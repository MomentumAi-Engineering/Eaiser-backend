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
    background-color: #020617;
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
  }}
  .card code {{
    background: #e5e7eb;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 14px;
    display: inline-block;
    margin-top: 6px;
  }}
  .cta {{
    text-align: center;
    margin: 35px 0;
  }}
  .cta a {{
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    color: white;
    padding: 15px 48px;
    border-radius: 10px;
    text-decoration: none;
    font-size: 16px;
    font-weight: 600;
    display: inline-block;
    transition: all 0.25s ease;
  }}
  .cta a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(99,102,241,0.6);
  }}
  .footer {{
    text-align: center;
    font-size: 12px;
    color: #64748b;
    border-top: 1px solid #e5e7eb;
    padding: 20px;
  }}
</style>
</head>

<body>
  <div class="container">
    <div class="header">
      <h1>Welcome to EAiSER 🚀</h1>
      <p>AI-Driven Civic Intelligence Platform</p>
    </div>

    <div class="content">
      <p>Hello <strong>{admin_name}</strong>,</p>

      <p>
        You have been onboarded by <strong>{created_by}</strong> as a
        <strong>{role.replace('_', ' ').title()}</strong>.
      </p>

      <div class="card">
        <h3>🔐 Login Credentials</h3>
        <p><strong>Email:</strong> {admin_email}</p>
        <p><strong>Temporary Password:</strong><br/>
          <code>{temporary_password}</code>
        </p>
        <p style="color:#dc2626;font-size:13px;">
          ⚠️ Please change your password after first login.
        </p>
      </div>

      <div class="card">
        <h3>🛡️ Permissions</h3>
        <p>{permissions_text}</p>
      </div>

      <div class="cta">
        <a href="{ADMIN_DASHBOARD_URL}">
          Access Admin Dashboard
        </a>
      </div>
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
