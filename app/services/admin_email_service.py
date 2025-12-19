"""
Admin Welcome Email Template
"""

async def send_admin_welcome_email(
    admin_email: str,
    admin_name: str,
    role: str,
    temporary_password: str,
    created_by: str
) -> bool:
    """
    Send welcome email to newly created admin with login credentials
    """
    try:
        subject = f"Welcome to EAiSER Admin Team - {role.replace('_', ' ').title()}"
        
        role_permissions = {
            "super_admin": "Full access - Manage team, assign issues, approve/decline reports",
            "admin": "Assign issues to team, approve/decline reports",
            "team_member": "Handle assigned issues, approve/decline assigned reports",
            "viewer": "View-only access to dashboard and reports"
        }
        
        permissions_text = role_permissions.get(role, "Standard admin access")
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                <h1 style="color: white; margin: 0;">Welcome to EAiSER Admin Team!</h1>
            </div>
            
            <div style="background-color: white; padding: 30px; border-radius: 0 0 10px 10px;">
                <p style="font-size: 16px; color: #333;">Hi <strong>{admin_name}</strong>,</p>
                
                <p style="font-size: 14px; color: #666; line-height: 1.6;">
                    You have been added to the EAiSER Admin Team by <strong>{created_by}</strong> 
                    with the role of <strong style="color: #667eea;">{role.replace('_', ' ').title()}</strong>.
                </p>
                
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #333;">Your Login Credentials</h3>
                    <p style="margin: 10px 0;"><strong>Email:</strong> {admin_email}</p>
                    <p style="margin: 10px 0;"><strong>Temporary Password:</strong> <code style="background: #e9ecef; padding: 5px 10px; border-radius: 4px; font-size: 14px;">{temporary_password}</code></p>
                    <p style="margin: 10px 0; color: #dc3545; font-size: 13px;">
                        ⚠️ Please change your password after first login
                    </p>
                </div>
                
                <div style="background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea;">
                    <h3 style="margin-top: 0; color: #333;">Your Permissions</h3>
                    <p style="margin: 0; color: #666;">{permissions_text}</p>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="http://localhost:5173/admin" 
                       style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 15px 40px; text-decoration: none; border-radius: 8px; 
                              font-weight: bold; font-size: 16px;">
                        Access Admin Dashboard
                    </a>
                </div>
                
                <p style="font-size: 13px; color: #999; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                    If you have any questions, please contact your team administrator.
                </p>
            </div>
        </div>
        """
        
        text_content = f"""
        Welcome to EAiSER Admin Team!
        
        Hi {admin_name},
        
        You have been added to the EAiSER Admin Team by {created_by} with the role of {role.replace('_', ' ').title()}.
        
        Your Login Credentials:
        Email: {admin_email}
        Temporary Password: {temporary_password}
        
        ⚠️ Please change your password after first login
        
        Your Permissions: {permissions_text}
        
        Login at: http://localhost:5173/admin
        
        If you have any questions, please contact your team administrator.
        """
        
        from services.email_service import send_email
        return await send_email(admin_email, subject, html_content, text_content)
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send welcome email to {admin_email}: {e}")
        return False
