"""
Enterprise-grade Admin Welcome Email Service
- Works in Development + Production
- Animated, professional HTML
- ENV-based frontend routing
"""

import os
import logging
from services.email_service import send_email, build_branded_email, EMAIL_STYLES

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
        # HTML EMAIL (branded shell)
        # ----------------------------
        role_title = role.replace('_', ' ').title()
        credential_card = (
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:24px 26px;margin:24px 0;">'
            '<h3 style="margin:0 0 16px;font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Security Credentials</h3>'
            '<div style="margin-bottom:14px;">'
            '<span style="font-size:12px;color:#94a3b8;display:block;margin-bottom:4px;">Access Email</span>'
            '<span style="font-size:16px;color:#1e293b;font-weight:500;">' + str(admin_email) + '</span>'
            '</div>'
            '<div>'
            '<span style="font-size:12px;color:#94a3b8;display:block;margin-bottom:4px;">Temporary Access Token</span>'
            '<div style="background:#ffffff;border:1px dashed #cbd5e1;padding:10px 15px;border-radius:8px;font-family:Monaco,Consolas,monospace;font-size:15px;color:#0f172a;font-weight:600;display:table;margin-top:5px;">' + str(temporary_password) + '</div>'
            '</div>'
            '<p style="margin:15px 0 0;font-size:11px;color:#ef4444;font-weight:600;">&#9888; PROTOCOL: For security compliance, you must update this password upon initial authentication.</p>'
            '</div>'
        )
        permissions_box = (
            '<div style="background:#fffbeb;border-left:4px solid #C8A84E;padding:18px 20px;border-radius:0 12px 12px 0;margin:24px 0;">'
            '<h4 style="margin:0 0 5px;color:#1e293b;font-size:15px;font-weight:700;">Privileged Scope</h4>'
            '<p style="margin:0;font-size:14px;color:#64748b;line-height:1.6;">' + str(permissions_text) + '</p>'
            '</div>'
        )
        inner_html = (
            '<h1 style="' + EMAIL_STYLES["h1"] + '">System Access Granted</h1>'
            '<div style="display:inline-block;padding:4px 12px;background:#f1f5f9;color:#475569;border-radius:6px;font-size:13px;font-weight:600;margin-bottom:18px;">Role: ' + role_title + '</div>'
            '<p style="' + EMAIL_STYLES["p"] + '">Hello <strong>' + str(admin_name) + '</strong>,</p>'
            '<p style="' + EMAIL_STYLES["p"] + '">You have been officially onboarded to the EAiSER administrative network by <strong>' + str(created_by) + '</strong>. Your account is now active and ready for deployment.</p>'
            + credential_card
            + permissions_box
            + '<div style="' + EMAIL_STYLES["btn_wrap"] + '">'
            '<a href="' + ADMIN_DASHBOARD_URL + '" style="' + EMAIL_STYLES["btn"] + '">Authenticate &amp; Launch Console</a>'
            '</div>'
        )
        html_content = build_branded_email(
            "Admin access granted",
            inner_html,
            preheader="Your EAiSER admin account is active — credentials inside.",
        )

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
