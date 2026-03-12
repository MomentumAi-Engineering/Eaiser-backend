"""
Enterprise-grade Admin Welcome Email Service
- Works in Development + Production
- Animated, professional HTML
- ENV-based frontend routing
"""

import os
import logging
from services.email_service import send_email

logger = logging.getLogger(__name__)

# -------------------------------------------------
# ENV CONFIG (DEV + PROD SAFE)
# -------------------------------------------------

ADMIN_DASHBOARD_URL = "https://www.eaiser.ai/admin"

# -------------------------------------------------
# MAIN SERVICE FUNCTION
# -------------------------------------------------

async def send_admin_welcome_email(
    admin_email: str,
    admin_name: str,
    role: str,
    temporary_password: str,
    created_by: str
) -> bool:
    """
    Sends a professional welcome email to newly created admins
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
        # HTML EMAIL (ENTERPRISE CONSOLE)
        # ----------------------------
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>EAiSER Administrative Onboarding</title>
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
    color: #ffffff;
    position: relative;
  }}
  .header h1 {{
    margin: 0;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f6c521;
  }}
  .header p {{
    margin: 8px 0 0;
    font-size: 14px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .content {{
    padding: 40px;
    color: #334155;
    font-size: 15px;
    line-height: 1.6;
  }}
  .greeting {{
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 12px;
  }}
  .access-badge {{
    display: inline-block;
    padding: 4px 12px;
    background: #f1f5f9;
    color: #475569;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 25px;
  }}
  .credential-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin: 30px 0;
  }}
  .credential-card h3 {{
    margin: 0 0 15px 0;
    font-size: 14px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .credential-row {{
    margin-bottom: 15px;
  }}
  .label {{
    font-size: 12px;
    color: #94a3b8;
    display: block;
    margin-bottom: 4px;
  }}
  .value {{
    font-size: 16px;
    color: #1e293b;
    font-weight: 500;
  }}
  .password-box {{
    background: #ffffff;
    border: 1px dashed #cbd5e1;
    padding: 10px 15px;
    border-radius: 8px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 15px;
    color: #0f172a;
    font-weight: 600;
    display: table;
    margin-top: 5px;
  }}
  .permissions-box {{
    border-left: 3px solid #f6c521;
    padding-left: 20px;
    margin: 30px 0;
  }}
  .permissions-box h4 {{
    margin: 0 0 5px 0;
    color: #1e293b;
  }}
  .permissions-box p {{
    margin: 0;
    font-size: 14px;
    color: #64748b;
  }}
  .cta-block {{
    text-align: center;
    margin: 40px 0 10px;
  }}
  .cta-button {{
    background: #1e293b;
    color: #ffffff !important;
    padding: 16px 40px;
    border-radius: 8px;
    text-decoration: none;
    font-size: 16px;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }}
  .footer {{
    padding: 30px 40px;
    background: #f8fafc;
    text-align: center;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    color: #94a3b8;
  }}
</style>
</head>
<body>
  <div class="wrapper">
    <div class="container">
      <div class="header">
        <h1>EAiSER Enterprise Console</h1>
        <p>Administrative Access Gateway</p>
      </div>

      <div class="content">
        <div class="greeting">System Access Granted</div>
        <div class="access-badge">Role: {role.replace('_', ' ').title()}</div>
        
        <p>Hello <strong>{admin_name}</strong>,</p>
        <p>You have been officially onboarded to the EAiSER administrative network by <strong>{created_by}</strong>. Your account is now active and ready for deployment.</p>

        <div class="credential-card">
          <h3>Security Credentials</h3>
          <div class="credential-row">
            <span class="label">Access Email</span>
            <span class="value">{admin_email}</span>
          </div>
          <div class="credential-row" style="margin-bottom: 0;">
            <span class="label">Temporary Access Token</span>
            <div class="password-box">{temporary_password}</div>
          </div>
          <p style="margin: 15px 0 0 0; font-size: 11px; color: #ef4444; font-weight: 600;">
            ⚠️ PROTOCOL: For security compliance, you must update this password upon initial authentication.
          </p>
        </div>

        <div class="permissions-box">
          <h4>Privileged Scope</h4>
          <p>{permissions_text}</p>
        </div>

        <div class="cta-block">
          <a href="{ADMIN_DASHBOARD_URL}" class="cta-button">
            Authenticate & Launch Console
          </a>
        </div>
      </div>

      <div class="footer">
        © 2026 MomntumAi · EAiSER Intelligence Platform<br/>
        This is an automated security transmission. Please do not reply.
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

You have been added by {created_by} as a {role.replace('_', ' ').title()}.

Login Credentials:
Email: {admin_email}
Temporary Password: {temporary_password}

Please change your password after first login.

Permissions:
{permissions_text}

Admin Dashboard:
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
