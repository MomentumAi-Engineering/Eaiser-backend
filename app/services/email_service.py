import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, Dict, Any
import base64
import asyncio
from datetime import datetime
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
    reply_to: Optional[str] = None,
    retry: bool = True,
    raw_attachments: Optional[List[Dict[str, Any]]] = None
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

    if reply_to:
        payload["ReplyTo"] = reply_to

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

    # Add raw pre-encoded attachments
    if raw_attachments:
        for attachment in raw_attachments:
            # Postmark expects keys: Name, Content, ContentType
            if all(k in attachment for k in ["Name", "Content", "ContentType"]):
                content_len = len(attachment.get("Content", ""))
                logger.info(f"📎 Adding raw attachment: {attachment.get('Name')} ({attachment.get('ContentType')}) - Size: {content_len} chars")
                payload["Attachments"].append(attachment)
                logger.debug(f"📎 Added raw attachment: {attachment.get('Name')}")

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
                return await send_email(
                    to_email, subject, html_content, text_content, 
                    attachments, embedded_images, 
                    reply_to=reply_to, retry=False, 
                    raw_attachments=raw_attachments
                )
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
        # 📧 Automated Email Communication Note
        # -------------------------------------------------------------
        try:
            # We need the real DB ID
            real_id = report.get("template_fields", {}).get("oid") or report.get("_id")
            
            if real_id:
                button_html = f"""
                <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0;">
                <div style="text-align: center; background-color: #f8fafc; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0;">
                    <h3 style="margin-top: 0; color: #1e293b; font-size: 18px;">📧 Official Communication</h3>
                    <p style="color: #64748b; margin-bottom: 25px; font-size: 14px;">To coordinate with the citizen regarding this incident, please <strong>reply directly to this email</strong>.</p>
                    
                    <p style="margin-top: 15px; font-size: 11px; color: #94a3b8; line-height: 1.5;">
                        <strong>Note:</strong> Your reply will be automatically routed to the reporter. You can request additional details or provide updates via this thread.<br>
                        Issue ID: #{real_id} • Automated Routing Active
                    </p>
                </div>
                """

                # Construct final HTML and Text bodies
                final_html = f"""
                <div style="font-family: 'Inter', system-ui, -apple-system, sans-serif; color: #1e293b; max-width: 600px; margin: 0 auto; border-radius: 24px; overflow: hidden; background-color: white; border: 1px solid #e2e8f0; box-shadow: 0 20px 50px rgba(0,0,0,0.1);">
                    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 40px 30px; text-align: center; border-bottom: 4px solid #fbbf24;">
                        <h2 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px;">🚀 EAiSER <span style="color:#fbbf24;">SMART ALERT</span></h2>
                        <div style="display: inline-block; padding: 4px 12px; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2); border-radius: 50px; font-size: 11px; font-weight: 700; color: #fbbf24; margin-top: 15px; text-transform: uppercase; letter-spacing: 1px;">
                            Official Intelligence Transmission
                        </div>
                    </div>
                    
                    <div style="padding: 40px 30px; font-size: 16px; color: #334155; line-height: 1.8;">
                        <div style="background: #f8fafc; border-radius: 16px; padding: 25px; border: 1px solid #f1f5f9; margin-bottom: 30px;">
                            {formatted_content.replace('\n', '<br>')}
                        </div>

                        {button_html}
                    </div>

                    <div style="padding: 25px 30px; background-color: #f8fafc; border-top: 1px solid #f1f5f9; text-align: center; font-size: 12px; color: #94a3b8;">
                        © 2026 EAiSER AI • Intelligent Civic Response Engine<br>
                        This is a privileged operational alert for verified authorities.
                    </div>
                </div>
                """
                
                final_text = f"{formatted_content}\n\nREPLY TO THIS EMAIL to communicate with the reporter (Issue ID: #{real_id})."
                
        except Exception as token_error:
            logger.error(f"⚠️ Failed to generate authority token: {token_error}")
            final_html = formatted_content.replace("\n", "<br>")
            final_text = formatted_content

        async def _send(to_email: str):
            # Use the Postmark inbound email address or the verified domain
            inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
            return await send_email(to_email, subject, final_html, final_text, reply_to=inbound_email)

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
    Notify the user that their report status has changed (Approved/Rejected) using new templates.
    """
    try:
        from services.mongodb_optimized_service import get_optimized_mongodb_service
        mongo = await get_optimized_mongodb_service()
        issue = await mongo.get_issue_by_id(issue_id) if mongo else {}
        
        # User name fallback
        user_name = "User"
        if issue and issue.get("reporter_name"):
            user_name = issue.get("reporter_name")
        elif user_email:
            user_name = user_email.split("@")[0].capitalize()

        # Issue Name / Description
        issue_type = "Issue"
        if issue and issue.get("issue_type"):
            issue_type = issue.get("issue_type").replace("_", " ").title()
        elif issue and issue.get("category"):
            issue_type = issue.get("category").replace("_", " ").title()
        
        # Subject prefix
        subject = f"Your EAiSER report – {issue_type} - Issue ID: {issue_id}"
        
        # Dashboard URL
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        dashboard_url = f"{frontend_url}/dashboard"

        # Premium Templates with Momntum AI branding
        logo_url = "https://www.momntumai.com/wp-content/uploads/2024/02/momntum-logo-white.png" # Example placeholder/typical URL
        
        # Base Styles
        base_html_start = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <style>
          body {{ background-color: #f8fafc; font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; color: #1e293b; }}
          .wrapper {{ padding: 40px 20px; }}
          .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }}
          .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 40px 40px; text-align: center; border-bottom: 4px solid #fbbf24; }}
          .logo {{ max-height: 40px; margin-bottom: 15px; }}
          .header h1 {{ margin: 0; color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }}
          .content {{ padding: 40px; font-size: 16px; line-height: 1.6; color: #475569; }}
          .greeting {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 20px; }}
          .highlight-box {{ background: #f1f5f9; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 0 8px 8px 0; margin: 25px 0; }}
          .highlight-box.approved {{ border-left-color: #22c55e; background: #f0fdf4; }}
          .highlight-box.declined {{ border-left-color: #ef4444; background: #fef2f2; }}
          .cta-container {{ text-align: center; margin: 40px 0 20px; }}
          .cta-btn {{ background: #fbbf24; color: #000000 !important; padding: 16px 36px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; box-shadow: 0 4px 6px rgba(251, 191, 36, 0.2); transition: all 0.2s; }}
          .footer {{ text-align: center; padding: 30px; background: #f8fafc; font-size: 13px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
        </style>
        </head>
        <body>
          <div class="wrapper">
            <div class="container">
              <div class="header">
                <!-- Using text fallback if logo fails to load -->
                <div style="color: #fbbf24; font-size: 28px; font-weight: 900; letter-spacing: 1px; margin-bottom: 10px;">Momntum<span style="color: #ffffff;">Ai</span></div>
                <h1>EAiSER Report Update</h1>
              </div>
              <div class="content">
                <div class="greeting">Hi {user_name},</div>
        """
        
        base_html_end = f"""
                <p>If you have any questions or additional information, please reply to this email and our support team will follow up as soon as possible.</p>
                
                <p style="margin-top: 30px; font-weight: 600; color: #1e293b;">
                  Best regards,<br>
                  The EAiSER Team
                </p>
                
                <div class="cta-container">
                  <a href="{dashboard_url}" class="cta-btn">Back to Dashboard</a>
                </div>
              </div>
              <div class="footer">
                © {datetime.utcnow().year} Momntum Ai. Empowering Communities.
              </div>
            </div>
          </div>
        </body>
        </html>
        """

        if status == "approved":
            # Extract authorities
            report = issue.get("report") or {}
            auths_data = report.get("responsible_authorities_or_parties") or report.get("available_authorities", [])
            auth_names = []
            for a in auths_data:
                if isinstance(a, dict) and a.get("name"):
                    auth_names.append(a.get("name"))
                elif isinstance(a, str):
                    auth_names.append(a)
            
            # Format authorities list e.g., "Authority X and Authority Y" or "the relevant authorities"
            if auth_names:
                if len(auth_names) > 1:
                    auth_display = ", ".join(auth_names[:-1]) + f" and {auth_names[-1]}"
                else:
                    auth_display = auth_names[0]
            else:
                auth_display = "the relevant authorities"

            html_content = base_html_start + f"""
                <p>We have completed the review of your report.</p>
                
                <div class="highlight-box approved">
                  <strong>Issue ID:</strong> {issue_id}<br><br>
                  Based on the information shared, we have successfully provided the details to <strong>{auth_display}</strong> for further action.
                </div>
            """ + base_html_end
            
            text_content = (
                f"Hi {user_name},\n\n"
                f"We have completed the review of your EAiSER report (Issue ID: {issue_id}).\n\n"
                f"Based on the information shared, we have provided the details to {auth_display}.\n\n"
                "If you have any questions or additional information, please reply to this email and our support team will follow up as soon as possible.\n\n"
                "Best regards,\n"
                "The EAiSER Team\n\n"
                f"Back to Dashboard: {dashboard_url}"
            )
            subject += "."  # The prompt has a dot at the end of the approved subject
        else:
            # Declined / Rejected
            html_content = base_html_start + f"""
                <p>We have completed the review of your report.</p>
                
                <div class="highlight-box">
                  <strong>Issue ID:</strong> {issue_id}<br><br>
                  At this time, we have not identified any issues that require notification to authorities. 
                </div>
            """ + base_html_end
            
            text_content = (
                f"Hi {user_name},\n\n"
                f"We have completed the review of your EAiSER report (Issue ID: {issue_id}).\n\n"
                "We have not identified any issues that require notification to authorities at this time.\n\n"
                "If you have any questions or further details to share, please reply to this email and our support team will follow up as soon as possible.\n\n"
                "Best regards,\n"
                "The EAiSER Team\n\n"
                f"Back to Dashboard: {dashboard_url}"
            )

        return await send_email(user_email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to notify user {user_email}: {e}", exc_info=True)
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
    background-color: #f8fafc;
    margin: 0;
    padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrapper {{
    background-color: #f8fafc;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 620px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
  }}
  .header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 45px 40px;
    text-align: center;
    position: relative;
  }}
  .header-accent {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, #fbbf24, #f59e0b);
  }}
  .header h1 {{
    margin: 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
  }}
  .header h1 span {{
    color: #fbbf24;
  }}
  .header p {{
    margin: 10px 0 0;
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
  }}
  .content {{
    padding: 40px;
    color: #334155;
    font-size: 16px;
    line-height: 1.7;
  }}
  .greeting {{
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
  }}
  .access-badge {{
    display: inline-block;
    padding: 5px 14px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #b45309;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 25px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .credential-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin: 30px 0;
  }}
  .credential-card h3 {{
    margin: 0 0 18px 0;
    font-size: 13px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
  }}
  .credential-row {{
    margin-bottom: 15px;
  }}
  .label {{
    font-size: 11px;
    color: #94a3b8;
    display: block;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }}
  .value {{
    font-size: 16px;
    color: #1e293b;
    font-weight: 600;
  }}
  .password-box {{
    background: #ffffff;
    border: 1px dashed #cbd5e1;
    padding: 12px 18px;
    border-radius: 10px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 16px;
    color: #0f172a;
    font-weight: 700;
    display: table;
    margin-top: 6px;
    letter-spacing: 1px;
  }}
  .permissions-box {{
    border-left: 4px solid #fbbf24;
    padding-left: 20px;
    margin: 30px 0;
    background: #fffbeb;
    padding: 20px;
    border-radius: 0 12px 12px 0;
  }}
  .permissions-box h4 {{
    margin: 0 0 6px 0;
    color: #1e293b;
    font-size: 15px;
    font-weight: 700;
  }}
  .permissions-box p {{
    margin: 0;
    font-size: 14px;
    color: #64748b;
    line-height: 1.6;
  }}
  .cta-block {{
    text-align: center;
    margin: 40px 0 15px;
  }}
  .cta-button {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #ffffff !important;
    padding: 16px 40px;
    border-radius: 12px;
    text-decoration: none;
    font-size: 16px;
    font-weight: 700;
    display: inline-block;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.2);
    letter-spacing: 0.3px;
  }}
  .footer {{
    padding: 30px 40px;
    background: #f8fafc;
    text-align: center;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    color: #94a3b8;
  }}
  .footer p {{
    margin: 5px 0;
  }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <div class="header-accent"></div>
        <h1>EAiSER <span>Console</span></h1>
        <p>Administrative Access Gateway</p>
      </div>

      <div class="content">
        <div class="greeting">System Access Granted</div>
        <div class="access-badge">Role: {role.replace('_', ' ').title()}</div>

        <p>Hello <strong>{admin_name}</strong>,</p>
        <p>You have been officially onboarded to the EAiSER administrative network by <strong>{created_by}</strong>. Your account is now active and ready for deployment.</p>

        <div class="credential-card">
          <h3>🔐 Security Credentials</h3>
          <div class="credential-row">
            <span class="label">Access Email</span>
            <span class="value">{admin_email}</span>
          </div>
          <div class="credential-row" style="margin-bottom: 0;">
            <span class="label">Temporary Access Token</span>
            <div class="password-box">{temporary_password}</div>
          </div>
          <p style="margin: 15px 0 0 0; font-size: 12px; color: #ef4444; font-weight: 600;">
            ⚠️ For security compliance, you must update this password upon initial authentication.
          </p>
        </div>

        <div class="permissions-box">
          <h4>🛡️ Privileged Scope</h4>
          <p>{permissions_text}</p>
        </div>

        <div class="cta-block">
          <a href="{ADMIN_DASHBOARD_URL}" class="cta-button">
            Launch Admin Console →
          </a>
        </div>

        <p style="text-align:center; color:#94a3b8; font-size:14px; margin-top:30px;">
          Welcome to the team. Let's make a difference.
        </p>
      </div>

      <div class="footer">
        <p>© 2026 MomntumAi · EAiSER Intelligence Platform</p>
        <p>This is an automated security transmission. Please do not reply.</p>
      </div>
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
# ✅ User Verification Email (Sleek & Actionable)
# --------------------------------------------------------------------

async def send_verification_email(user_email: str, user_name: str, token: str) -> bool:
    """
    Sends a verification email to new users.
    """
    try:
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
            
        verification_link = f"{frontend_url}/verify-email?token={token}"
        subject = "Verify your EAiSER AI Account"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ background-color: #f8fafc; font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
  .wrapper {{ padding: 40px 20px; background-color: #f8fafc; }}
  .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1); }}
  .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 45px 40px; text-align: center; position: relative; }}
  .header-accent {{ position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(to right, #fbbf24, #f59e0b); }}
  .header h1 {{ margin: 0; color: #fbbf24; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }}
  .header p {{ margin: 10px 0 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }}
  .content {{ padding: 40px; color: #334155; line-height: 1.7; font-size: 16px; }}
  .content h2 {{ color: #0f172a; font-size: 22px; font-weight: 700; margin-top: 0; }}
  .cta {{ text-align: center; margin: 35px 0; }}
  .cta a {{ background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: #000000 !important; padding: 16px 40px; border-radius: 12px; text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; box-shadow: 0 10px 25px rgba(251, 191, 36, 0.25); }}
  .link-fallback {{ font-size: 13px; color: #64748b; word-break: break-all; background: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin-top: 20px; }}
  .expiry {{ display: inline-block; padding: 6px 14px; background: #fef2f2; color: #dc2626; border-radius: 8px; font-size: 13px; font-weight: 600; margin-top: 10px; }}
  .footer {{ text-align: center; padding: 30px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; background: #f8fafc; }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <div class="header-accent"></div>
        <h1>EAiSER AI</h1>
        <p>Account Verification</p>
      </div>
      <div class="content">
        <h2>Welcome to the Mission, {user_name}!</h2>
        <p>Please verify your email address to activate your account and start reporting civic issues with AI precision.</p>
        <div class="cta">
          <a href="{verification_link}">Verify My Account →</a>
        </div>
        <div class="link-fallback">
          Or copy this link into your browser:<br/>{verification_link}
        </div>
        <p style="text-align: center;"><span class="expiry">⏰ This link will expire in 24 hours.</span></p>
      </div>
      <div class="footer">© 2026 EAiSER AI · Automated Civic Intelligence Network</div>
    </div>
  </div>
</body>
</html>
"""
        text_content = f"Welcome to EAiSER AI, {user_name}!\n\nPlease verify your email: {verification_link}"
        
        return await send_email(user_email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

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
        2026 EAiSER AI · Automated Civic Network<br>
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
# --------------------------------------------------------------------
# ✅ Password Reset Email
# --------------------------------------------------------------------

async def send_password_reset_email(email: str, token: str) -> bool:
    """
    Sends a secure password reset email with a 15-minute expiry warning.
    """
    try:
        # Determine Frontend URL
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        if frontend_url.endswith("/"):
            frontend_url = frontend_url[:-1]

        reset_link = f"{frontend_url}/reset-password?token={token}"
        subject = "Reset Your EAiSER Password"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ background-color: #f8fafc; font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
  .wrapper {{ padding: 40px 20px; background-color: #f8fafc; }}
  .container {{ max-width: 560px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1); }}
  .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 40px; text-align: center; position: relative; }}
  .header-accent {{ position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(to right, #fbbf24, #f59e0b); }}
  .header h2 {{ color: #ffffff; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }}
  .header p {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; margin: 10px 0 0; font-weight: 600; }}
  .content {{ padding: 40px; color: #475569; line-height: 1.7; font-size: 16px; }}
  .cta {{ text-align: center; margin: 35px 0; }}
  .cta a {{ display: inline-block; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: #0f172a !important; padding: 16px 40px; border-radius: 12px; text-decoration: none; font-weight: 700; font-size: 16px; box-shadow: 0 10px 25px rgba(251, 191, 36, 0.25); }}
  .safety-note {{ background: #f8fafc; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; font-size: 14px; color: #64748b; margin-top: 25px; }}
  .expiry {{ display: inline-block; padding: 5px 12px; background: #fef2f2; color: #dc2626; border-radius: 8px; font-size: 12px; font-weight: 600; }}
  .footer {{ padding: 25px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; background: #f8fafc; }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <div class="header-accent"></div>
        <h2>🔒 Secure Password Reset</h2>
        <p>Identity Protection</p>
      </div>
      <div class="content">
        <p>We received a request to reset your EAiSER account password. Click the button below to choose a new one.</p>
        <p><span class="expiry">⏰ This link expires in 15 minutes</span></p>
        <div class="cta">
          <a href="{reset_link}">Reset My Password →</a>
        </div>
        <div class="safety-note">
          🛡️ If you didn't request this, you can safely ignore this email. Your password will remain unchanged.
        </div>
      </div>
      <div class="footer">© 2026 EAiSER AI · Secure Identity Protection</div>
    </div>
  </div>
</body>
</html>
"""
        text_content = f"Reset your EAiSER password: {reset_link}\n\nThis link will expire in 15 minutes."

        return await send_email(email, subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email}: {e}")
        return False

async def send_tos_email(email: str, name: str) -> bool:
    """Send a copy of the Terms of Service to the user upon acceptance."""
    try:
        subject = "Your Accepted Terms: EAiSER & MomntumAi"
        
        frontend_url = os.getenv("FRONTEND_URL", "https://www.eaiser.ai")
        if frontend_url.endswith("/"): frontend_url = frontend_url[:-1]
        
        # Path to the final Terms PDF
        tos_pdf = r"c:/Users/chris/OneDrive/Desktop/MomntumAi/momentum-frontend/public/TERMSand.pdf"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; background-color: #f8fafc; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
    .wrapper {{ padding: 40px 20px; background-color: #f8fafc; }}
    .container {{ max-width: 560px; margin: 0 auto; background-color: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
    .header {{ background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); padding: 60px 30px; text-align: center; border-bottom: 1px solid #e2e8f0; position: relative; }}
    .header-accent {{ position: absolute; top: 0; left: 0; right: 0; height: 5px; background: linear-gradient(to right, #fbbf24, #f59e0b); }}
    .header h1 {{ color: #0f172a; font-size: 26px; margin: 0; font-weight: 800; letter-spacing: -0.5px; }}
    .content {{ padding: 45px 40px; }}
    p {{ color: #475569; font-size: 16px; line-height: 1.8; margin-bottom: 25px; }}
    .check-badge {{ 
      display: inline-flex; 
      align-items: center; 
      gap: 10px; 
      background: #ecfdf5; 
      border: 1px solid #10b981; 
      padding: 14px 24px; 
      border-radius: 50px; 
      color: #065f46; 
      font-weight: 700; 
      font-size: 15px; 
      margin: 30px 0;
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }}
    .check-icon {{ font-size: 18px; margin-right: 8px; }}
    .tos-link {{ color: #3b82f6; text-decoration: none; font-weight: 600; border-bottom: 2px solid rgba(59,130,246,0.1); transition: all 0.2s; }}
    .tos-link:hover {{ color: #2563eb; border-bottom-color: #2563eb; }}
    .footer {{ padding: 30px; text-align: center; font-size: 13px; color: #94a3b8; border-top: 1px solid #e2e8f0; background: #f8fafc; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <div class="header-accent"></div>
        <h1>Terms of Service Confirmation</h1>
      </div>
      <div class="content">
        <p>Hi {name or 'User'},</p>
        <div class="check-badge"><span class="check-icon">✅</span> Terms of Service Accepted</div>
        <p>Thank you for using EAiSER AI. This email confirms that you have reviewed and accepted our Terms of Service and Privacy Policy.</p>
        <p>As requested, we have attached a copy of these legal documents for your records. We're thrilled to have you on board. Together, we're building smarter, safer communities.</p>
        <p style="font-weight: 700; color: #0f172a; margin-top: 40px;">— The EAiSER & MomntumAi Team</p>
      </div>
      <div class="footer">© {datetime.utcnow().year} EAiSER AI · Intelligent Civic Response</div>
    </div>
  </div>
</body>
</html>
"""
        text_content = f"Hi {name or 'User'},\n\nThank you for using EAiSER.Ai. This email serves as confirmation that you have agreed to our Terms of Service. We have attached a copy for your records.\n\nThanks,\nThe EAiSER & MomntumAi Team"

        attachments = [tos_pdf] if os.path.exists(tos_pdf) else None
        return await send_email(email, subject, html_content, text_content, attachments=attachments)
    except Exception as e:
        logger.error(f"Failed to send TOS email to {email}: {e}")
        return False

# --------------------------------------------------------------------
# ✅ Government Portal Welcome Email
# --------------------------------------------------------------------

GOV_PORTAL_URL = "https://gov.eaiser.ai/login"

async def send_gov_welcome_email(
    email: str,
    name: str,
    department: str,
    city: str,
    zip_code: str,
    temporary_password: str
) -> bool:
    """
    Sends a professional welcome email to newly created Government Official accounts.
    """
    try:
        subject = f"Security Clearance Granted — EAiSER Government Portal ({department})"
        
        # Style inspired by the admin welcome email but tailored for Gov/Civic feel
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ background-color: #f8fafc; margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; -webkit-font-smoothing: antialiased; }}
  .wrapper {{ background-color: #f8fafc; padding: 40px 20px; }}
  .container {{ max-width: 620px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
  .header {{ background: #000000; padding: 45px 40px; text-align: center; border-bottom: 4px solid #fbbf24; }}
  .header h1 {{ margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; color: #ffffff; }}
  .header h1 span {{ color: #fbbf24; }}
  .header p {{ margin: 10px 0 0; font-size: 13px; color: #71717a; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }}
  .content {{ padding: 40px 50px; color: #334155; font-size: 16px; line-height: 1.7; }}
  .greeting {{ font-size: 20px; font-weight: 800; color: #000000; margin-bottom: 8px; }}
  .dept-badge {{ display: inline-block; padding: 5px 14px; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2); color: #d97706; border-radius: 50px; font-size: 11px; font-weight: 800; margin-bottom: 25px; text-transform: uppercase; }}
  .credential-card {{ background: #fafafa; border: 1px solid #f1f1f1; border-radius: 16px; padding: 30px; margin: 30px 0; }}
  .label {{ font-size: 10px; color: #999; display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; }}
  .value {{ font-size: 15px; color: #111; font-weight: 700; }}
  .password-box {{ background: #ffffff; border: 1px dashed #ddd; padding: 12px 18px; border-radius: 10px; font-family: 'Consolas', monospace; font-size: 17px; color: #000; font-weight: 800; display: table; margin-top: 8px; }}
  .scope-box {{ background: #f0fdf4; border: 1px solid #dcfce7; padding: 20px; border-radius: 12px; margin-bottom: 30px; }}
  .scope-box h4 {{ margin: 0 0 6px 0; color: #166534; font-size: 14px; font-weight: 800; }}
  .scope-box p {{ margin: 0; font-size: 13px; color: #15803d; font-weight: 500; }}
  .cta-block {{ text-align: center; margin-top: 40px; }}
  .cta-button {{ background: #000; color: #fff !important; padding: 16px 40px; border-radius: 12px; text-decoration: none; font-size: 15px; font-weight: 800; display: inline-block; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
  .footer {{ padding: 30px; background: #fafafa; text-align: center; border-top: 1px solid #f1f1f1; font-size: 12px; color: #a1a1aa; }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <h1>EA<span>i</span>SER <span>Gov</span></h1>
        <p>Authority Command Center</p>
      </div>
      <div class="content">
        <div class="greeting">Official Clearance Granted</div>
        <div class="dept-badge">{department} — {city}</div>
        <p>Hello <strong>{name}</strong>,</p>
        <p>Your official credentials for the EAiSER Government Portal have been provisioned by the EAiSER Global Operations team. You now have access to real-time citizen report routing and AI-driven department insights.</p>
        
        <div class="credential-card">
          <div style="margin-bottom: 20px;">
            <span class="label">Portal ID / Email</span>
            <span class="value">{email.lower()}</span>
          </div>
          <div>
            <span class="label">Temporary Security Password</span>
            <div class="password-box">{temporary_password}</div>
          </div>
        </div>

        <div class="scope-box">
          <h4>🛰️ Operation Scope</h4>
          <p>You are authorized to monitor and manage reports for the <strong>{department}</strong> within ZIP context <strong>{zip_code}</strong>. This dashboard is strictly for official use.</p>
        </div>

        <div class="cta-block">
          <a href="{GOV_PORTAL_URL}" class="cta-button">Access Gov Dashboard →</a>
        </div>
      </div>
      <div class="footer">
        <p>© 2026 EAiSER AI • Intelligent Civic Infrastructure</p>
      </div>
    </div>
  </div>
</body>
</html>
"""
        text_content = f"""
Official Clearance Granted — EAiSER Gov

Hello {name},
Your credentials for the EAiSER Government Portal are ready.

Department: {department}
ZIP Context: {zip_code}

Portal ID: {email.lower()}
Temporary Password: {temporary_password}

Access the dashboard at: {GOV_PORTAL_URL}
"""
        return await send_email(email.lower(), subject, html_content, text_content)
    except Exception as e:
        logger.error(f"Failed to send gov welcome email: {e}")
        return False
