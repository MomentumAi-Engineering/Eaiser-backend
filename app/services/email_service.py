import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, Dict, Any
import base64
import asyncio
from fastapi import HTTPException
import requests

# --------------------------------------------------------------------
# ✅ Environment & Logger Setup
# --------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger("email_service")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------
# ✅ Core Async Email Sender with Improved Error Handling
# --------------------------------------------------------------------
async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None,
    retry: bool = True
) -> bool:
    """
    Sends an email via Postmark API with inline images and attachments.
    Uses asyncio.to_thread to keep the event loop non-blocking.
    """
    email_user = os.getenv("EMAIL_USER", "alert@momntumai.com")
    postmark_token = os.getenv("POSTMARK_API_TOKEN")
    dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"

    if dry_run:
        logger.info(f"🧪 Email dry-run enabled. Would send to {to_email} subject='{subject}'.")
        return True

    if not postmark_token:
        logger.error("❌ Missing POSTMARK_API_TOKEN environment variable.")
        return False

    # Build Postmark payload
    payload = {
        "From": email_user,
        "To": to_email,
        "Subject": subject,
        "HtmlBody": html_content,
        "TextBody": text_content,
        "Attachments": []
    }

    # Add inline images to Attachments
    if embedded_images:
        for cid, base64_data, mime_type in embedded_images:
            try:
                if not base64_data:
                    logger.warning(f"⚠️ Skipping empty embedded image {cid}")
                    continue
                payload["Attachments"].append({
                    "Name": f"{cid}.{mime_type.split('/')[-1]}",
                    "Content": base64_data,
                    "ContentType": mime_type,
                    "ContentID": f"cid:{cid}"
                })
                logger.debug(f"🖼️ Embedded image {cid} added.")
            except Exception as e:
                logger.error(f"⚠️ Failed to prepare inline image {cid}: {e}")

    # Add attachments to Attachments
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                # Postmark requires base64 content
                encoded = base64.b64encode(data).decode()
                payload["Attachments"].append({
                    "Name": os.path.basename(file_path),
                    "Content": encoded,
                    "ContentType": "application/octet-stream"
                })
                logger.debug(f"📎 Attached file: {file_path}")
            except Exception as e:
                logger.error(f"⚠️ Failed to prepare attachment {file_path}: {e}")

    def _do_send():
        url = "https://api.postmarkapp.com/email"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Postmark-Server-Token": postmark_token
        }
        logger.info(f"📤 Sending Postmark email FROM {email_user} TO {to_email}")
        return requests.post(url, headers=headers, json=payload, timeout=15)

    try:
        response = await asyncio.to_thread(_do_send)
        
        if response.status_code == 200:
            logger.info(f"✅ Email successfully sent to {to_email} via Postmark")
            return True
        else:
            try:
                error_data = response.json()
                msg = error_data.get('Message', 'Unknown Postmark error')
            except:
                msg = response.text
            
            logger.warning(f"⚠️ Postmark API returned {response.status_code}: {msg}")
            
            if retry and response.status_code != 401:
                logger.info("🔁 Retrying once after 2 seconds...")
                await asyncio.sleep(2)
                return await send_email(to_email, subject, html_content, text_content, attachments, embedded_images, retry=False)
            return False

    except Exception as e:
        logger.error(f"❌ Unexpected error sending email to {to_email} via Postmark: {e}")
        return False


# --------------------------------------------------------------------
# ✅ Synchronous Wrapper for Celery/Testing
# --------------------------------------------------------------------
def send_email_sync(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None
) -> bool:
    """Run async send_email synchronously (for local testing or Celery)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            send_email(to_email, subject, html_content, text_content, attachments, embedded_images)
        )
    except Exception as e:
        logger.error(f"❌ Sync wrapper failed: {e}")
        return False
    finally:
        loop.close()


# --------------------------------------------------------------------
# ✅ EmailService Class for Issue Notifications
# --------------------------------------------------------------------
class EmailService:
    async def send_issue_notification(self, authorities: List[Dict[str, Any]], issue_data: Dict[str, Any], issue_id: str) -> bool:
        """Send AI-detected issue notification emails to relevant authorities."""
        try:
            subject = f"New Issue Report #{issue_data.get('report_id', issue_id)}"
            text_content = (
                f"Issue ID: {issue_id}\n"
                f"Type: {issue_data.get('issue_type')}\n"
                f"Severity: {issue_data.get('severity')}\n"
                f"Description: {issue_data.get('description', 'N/A')}\n"
                f"Location: {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})\n"
            )
            html_content = (
                f"<h3>🛰️ New Issue Report Detected</h3>"
                f"<p><strong>Issue ID:</strong> {issue_id}</p>"
                f"<p><strong>Type:</strong> {issue_data.get('issue_type')}</p>"
                f"<p><strong>Severity:</strong> {issue_data.get('severity')}</p>"
                f"<p><strong>Description:</strong> {issue_data.get('description', 'N/A')}</p>"
                f"<p><strong>Location:</strong> {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})</p>"
            )

            sent_any = False
            for auth in authorities:
                to_email = auth.get('email') or auth.get('contact_email')
                if not to_email:
                    logger.warning(f"⚠️ Skipping authority without email: {auth}")
                    continue
                ok = await send_email(to_email, subject, html_content, text_content)
                sent_any = sent_any or ok

            logger.info(f"📬 Issue email notifications sent: {sent_any}")
            return sent_any

        except Exception as e:
            logger.error(f"❌ Issue notification failed: {e}", exc_info=True)
            return False


def get_email_service() -> EmailService:
    """Return a reusable instance of EmailService."""
    return EmailService()


# --------------------------------------------------------------------
# ✅ Send AI-Formatted Alert to Authorities (EAiSER Alert)
# --------------------------------------------------------------------
async def send_formatted_ai_alert(report: Dict[str, Any], background: bool = True) -> Dict[str, Any]:
    """
    Send the AI-generated formatted EAiSER alert to authorities.
    Uses the 'formatted_report' field from the AI report.
    """
    try:
        env = os.getenv("ENV", "development").lower()
        dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"
        formatted_content = report.get("formatted_report", "")
        issue_type = report.get("issue_overview", {}).get("type", "Issue")
        report_id = report.get("template_fields", {}).get("oid", "N/A")
        priority = report.get("template_fields", {}).get("priority", "N/A")
        subject = f"EAiSER Alert – {issue_type} (Priority: {priority}, ID: {report_id})"

        authorities = report.get("responsible_authorities_or_parties") or report.get("available_authorities") or []
        recipients = [a.get("email") for a in authorities if isinstance(a, dict) and a.get("email")]

        if not recipients:
            logger.warning("⚠️ No recipients found in AI report.")
            return {"status": "no_recipients", "recipients": []}

        # Only skip if dry_run is explicitly enabled
        if dry_run:
            logger.info(
                f"🧪 Email dry-run active. Skipping send to {len(recipients)} recipients"
            )
            return {"status": "dry_run", "recipients": recipients}

        # -------------------------------------------------------------
        # 🔥 Generate Secure Authority Action Link (Step 1 Strategy)
        # -------------------------------------------------------------
        try:
            # We need the real DB ID for the token
            real_id = report.get("template_fields", {}).get("oid") or report.get("_id")
            
            if real_id:
                # Import here to avoid circular dependency
                try:
                    from app.routes.authority_action import create_authority_token
                except ImportError:
                    from routes.authority_action import create_authority_token
                
                token = create_authority_token(str(real_id))
                
                # Determine Frontend URL
                frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
                # Fallback for local dev if not set
                if not os.getenv("FRONTEND_URL") and os.getenv("ENV") == "development":
                    frontend_url = "http://localhost:5173"

                action_link = f"{frontend_url}/authority-action?token={token}"
                
                logger.info(f"🔗 Generated Authority Action Link for {real_id}")

                button_html = f"""
                <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0;">
                <div style="text-align: center; background-color: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <h3 style="margin-top: 0; color: #1e293b;">⚡ Official Action Required</h3>
                    <p style="color: #64748b; margin-bottom: 20px;">Use the secure link below to update the status of this issue directly.</p>
                    
                    <a href="{action_link}" style="background-color: #2563eb; color: white; padding: 16px 32px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 16px; display: inline-block; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);">
                        View & Manage Issue
                    </a>
                    
                    <p style="margin-top: 15px; font-size: 12px; color: #94a3b8;">
                        Secure Access • No Login Required • Expires in 7 Days
                    </p>
                </div>
                """
                
                formatted_content += button_html
                
        except Exception as token_error:
            logger.error(f"⚠️ Failed to generate authority token: {token_error}")


        async def _send(to_email: str):
            html = formatted_content.replace("\n", "<br>")
            return await send_email(to_email, subject, html, formatted_content)

        if background:
            for r in recipients:
                asyncio.create_task(_send(r))
            logger.info(f"📤 Dispatched EAiSER Alert to {len(recipients)} authorities (background mode)")
            return {"status": "dispatched", "recipients": recipients}
        else:
            sent, failed = [], []
            for r in recipients:
                ok = await _send(r)
                (sent if ok else failed).append(r)
            return {"status": "completed", "sent": sent, "failed": failed}

    except Exception as e:
        logger.error(f"❌ send_formatted_ai_alert failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

# --------------------------------------------------------------------
# ✅ Notify User Status Change
# --------------------------------------------------------------------
async def notify_user_status_change(user_email: str, issue_id: str, status: str, notes: Optional[str] = None) -> bool:
    """
    Notify the user that their report status has changed (Approved/Rejected).
    """
    try:
        subject = f"Update on your Report #{issue_id}"
        
        status_color = "green" if status == "approved" else "red"
        status_display = "APPROVED" if status == "approved" else "DECLINED"
        
        html_content = f"""
        <h2>Report Status Update</h2>
        <p>Your report (ID: <strong>{issue_id}</strong>) has been updated.</p>
        <p>New Status: <strong style="color: {status_color}">{status_display}</strong></p>
        """
        
        if notes:
            html_content += f"<p><strong>Admin Notes:</strong> {notes}</p>"
            
        html_content += "<p>Thank you for using EAiSER Ai.</p>"
        
        text_content = f"Your report {issue_id} has been {status_display}.\n"
        if notes:
            text_content += f"Notes: {notes}\n"
            
        return await send_email(user_email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to notify user {user_email}: {e}")
        return False


# --------------------------------------------------------------------
# ✅ Admin Welcome Email (Animated & Professional)
# --------------------------------------------------------------------

ADMIN_DASHBOARD_URL = "https://www.eaiser.ai/admin"

async def send_admin_welcome_email(
    admin_email: str,
    admin_name: str,
    role: str,
    temporary_password: str,
    created_by: str
) -> bool:
    """
    Sends a professional, animated welcome email to newly created admins.
    """
    try:
        # ----------------------------
        # Role Permissions Mapping
        # ----------------------------
        role_permissions = {
            "super_admin": "Full system access — manage admins, assign issues, approve or decline reports.",
            "admin": "Manage team members, assign issues, approve or decline reports.",
            "team_member": "Handle assigned issues and review reports.",
            "viewer": "Read-only access to dashboards and reports."
        }

        permissions_text = role_permissions.get(
            role,
            "Standard administrative access."
        )

        subject = f"Welcome to EAiSER — {role.replace('_', ' ').title()} Access Granted"

        # ----------------------------
        # HTML EMAIL (ANIMATED + PRO)
        # ----------------------------
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Welcome to EAiSER</title>
<style>
  body {{
    background-color: #020617; /* Dark slate background */
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', Arial, sans-serif;
  }}
  .container {{
    max-width: 620px;
    margin: 40px auto;
    background: #ffffff;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 25px 70px rgba(0,0,0,0.45);
    animation: slideUp 0.9s ease-out;
  }}
  @keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(25px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .header {{
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    padding: 35px;
    text-align: center;
    color: #ffffff;
  }}
  .header h1 {{
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }}
  .header p {{
    margin: 10px 0 0;
    font-size: 16px;
    opacity: 0.9;
  }}
  .content {{
    padding: 35px;
    color: #334155;
    font-size: 15px;
    line-height: 1.7;
  }}
  .card {{
    background: #f8fafc;
    border-radius: 10px;
    padding: 22px;
    margin: 25px 0;
    border-left: 4px solid #6366f1;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}
  .card h3 {{
    margin-top: 0;
    color: #1e293b;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .card code {{
    background: #e5e7eb;
    padding: 8px 12px;
    border-radius: 6px;
    font-family: 'Consolas', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #334155;
    display: inline-block;
    margin-top: 6px;
    letter-spacing: 1px;
  }}
  .cta {{
    text-align: center;
    margin: 35px 0;
  }}
  .cta a {{
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    color: white;
    padding: 16px 48px;
    border-radius: 50px;
    text-decoration: none;
    font-size: 16px;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
    transition: all 0.25s ease;
  }}
  .cta a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(79, 70, 229, 0.5);
  }}
  .footer {{
    text-align: center;
    font-size: 12px;
    color: #64748b;
    background: #f1f5f9;
    padding: 20px;
    border-top: 1px solid #e2e8f0;
  }}
</style>
</head>

<body>
  <div class="container">
    <div class="header">
      <h1>EAiSER Access</h1>
      <p>AI-Driven Civic Intelligence Platform</p>
    </div>

    <div class="content">
      <p>Hello <strong>{admin_name}</strong>,</p>

      <p>
        You have been officially onboarded by <strong>{created_by}</strong> as a
        <strong>{role.replace('_', ' ').title()}</strong>.
      </p>

      <div class="card">
        <h3>🔐 Login Credentials</h3>
        <p><strong>Email:</strong> {admin_email}</p>
        <p><strong>Temporary Password:</strong><br/>
          <code>{temporary_password}</code>
        </p>
        <p style="color:#ef4444;font-size:13px; font-weight:500;">
          ⚠️ For security, please change your password immediately after logging in.
        </p>
      </div>

      <div class="card">
        <h3>🛡️ Your Permissions</h3>
        <p>{permissions_text}</p>
      </div>

      <div class="cta">
        <a href="{ADMIN_DASHBOARD_URL}">
          Launch Admin Dashboard
        </a>
      </div>
      
      <p style="text-align:center; color:#64748b; margin-top:30px;">
        Welcome to the team. Let's make a difference.
      </p>
    </div>

    <div class="footer">
      © {created_by} · EAiSER Platform<br/>
      Secure · Scalable · Intelligent
    </div>
  </div>
</body>
</html>
"""

        # ----------------------------
        # TEXT EMAIL (Fallback)
        # ----------------------------
        text_content = f"""
Welcome to EAiSER

Hi {admin_name},

You have been onboarded by {created_by} as a {role.replace('_', ' ').title()}.

Your Login Credentials:
-----------------------
Email: {admin_email}
Temporary Password: {temporary_password}

(Please change your password after first login)

Your Permissions:
-----------------
{permissions_text}

Access Admin Dashboard:
{ADMIN_DASHBOARD_URL}

— EAiSER Platform
"""

        # ----------------------------
        # SEND EMAIL
        # ----------------------------
        return await send_email(
            to_email=admin_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )

    except Exception as e:
        logger.error(f"❌ Failed to send admin welcome email to {admin_email}: {e}")
        return False


# --------------------------------------------------------------------
# ✅ User Welcome Email (Sleek & Explanatory)
# --------------------------------------------------------------------

async def send_user_welcome_email(user_email: str, user_name: str) -> bool:
    """
    Sends a premium, high-converting welcome email to new users.
    Focuses on rich aesthetics, animations, and community impact.
    """
    try:
        subject = "Welcome to EAiSER AI – Your Journey to a Smarter Community Begins 🚀"
        
        # ----------------------------
        # HTML EMAIL (PREMIUM MODERN DESIGN)
        # ----------------------------
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{
    background-color: #f8fafc;
    font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    color: #1e293b;
  }}
  .wrapper {{
    background-color: #f8fafc;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 600px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
  }}
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 60px 40px;
    text-align: center;
    position: relative;
  }}
  .header h1 {{
    margin: 0;
    color: #fbbf24; /* Amber 400 */
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    text-transform: uppercase;
  }}
  .header .badge {{
    display: inline-block;
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
    color: #fbbf24;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 15px;
  }}
  .content {{
    padding: 40px;
  }}
  .greeting {{
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 20px;
  }}
  .intro {{
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 40px;
    color: #475569;
  }}
  .feature-grid {{
    margin: 40px 0;
  }}
  .feature-item {{
    display: flex;
    align-items: flex-start;
    margin-bottom: 25px;
    padding: 20px;
    background: #f1f5f9;
    border-radius: 12px;
    transition: transform 0.2s ease;
  }}
  .icon {{
    font-size: 24px;
    margin-right: 15px;
    background: #fff;
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  }}
  .feature-text {{
    flex: 1;
  }}
  .feature-title {{
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
    font-size: 16px;
  }}
  .feature-desc {{
    font-size: 14px;
    color: #64748b;
    line-height: 1.5;
  }}
  .cta-container {{
    text-align: center;
    margin: 50px 0 20px;
  }}
  .cta-btn {{
    background: #1e293b;
    color: #ffffff !important;
    padding: 18px 45px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 700;
    font-size: 16px;
    display: inline-block;
    box-shadow: 0 10px 25px rgba(30, 41, 59, 0.2);
    transition: all 0.3s ease;
  }}
  .cta-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(30, 41, 59, 0.3);
  }}
  .footer {{
    text-align: center;
    padding: 40px;
    background: #f8fafc;
    font-size: 13px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
  }}
  .social-links {{
    margin-top: 20px;
  }}
  .social-links a {{
    margin: 0 10px;
    color: #94a3b8;
    text-decoration: none;
  }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <h1>EAiSER AI</h1>
        <div class="badge">BETA ACCESS GRANTED</div>
      </div>
      
      <div class="content">
        <div class="greeting">Welcome to the inner circle, {user_name}!</div>
        <p class="intro">
          You've just unlocked the most powerful tool for community transformation. EAiSER isn't just an app—it's an AI-driven mission to eliminate civic hazards and build a more responsive world, one report at a time.
        </p>

        <div class="feature-grid">
          <div class="feature-item">
            <div class="icon">📸</div>
            <div class="feature-text">
              <div class="feature-title">Impact via Imagery</div>
              <div class="feature-desc">Snap a photo and let our Neural Engine handle the rest. We identify complexity, severity, and root causes instantly.</div>
            </div>
          </div>

          <div class="feature-item">
            <div class="icon">⚡</div>
            <div class="feature-text">
              <div class="feature-title">Autonomous Routing</div>
              <div class="feature-desc">Say goodbye to bureaucratic hurdles. EAiSER routes your data directly to the precise authority responsible for action.</div>
            </div>
          </div>

          <div class="feature-item">
            <div class="icon">📈</div>
            <div class="feature-text">
              <div class="feature-title">Real-time Visibility</div>
              <div class="feature-desc">Monitor the status of your reports on a live dashboard. See exactly when officials receive, review, and resolve issues.</div>
            </div>
          </div>
        </div>

        <div class="cta-container">
          <a href="https://www.eaiser.ai/dashboard" class="cta-btn">Initialize My Dashboard</a>
        </div>
        
        <p style="text-align:center; color:#94a3b8; font-size:14px; margin-top:30px;">
          Together, we're building the future of civic technology.
        </p>
      </div>

      <div class="footer">
        <strong>MomntumAi LLC</strong><br>
        2025 EAiSER AI · Automated Civic Network<br>
        <div class="social-links">
          <a href="#">Terms</a> • <a href="#">Privacy</a> • <a href="#">Support</a>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
        # ----------------------------
        # TEXT FALLBACK
        # ----------------------------
        text_content = f"""
Welcome to EAiSER AI, {user_name}!

You've just joined the most powerful network for community transformation.

What's Next?
1. Snap & Report: Our Neuro Engine analyzes issues instantly.
2. Autonomous Routing: We send data directly to the right authorities.
3. Real-time Tracking: Monitor progress on your live dashboard.

Get Started: https://www.eaiser.ai/dashboard

Together, let's build smarter communities.
- Team EAiSER
"""
        return await send_email(user_email, subject, html_content, text_content)

    except Exception as e:
        logger.error(f"❌ Failed to send user welcome email to {user_email}: {e}")
        return False
