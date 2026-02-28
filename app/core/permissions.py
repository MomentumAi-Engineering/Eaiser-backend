
# Role-Based Access Control Matrix
# Matches frontend src/utils/permissions.js

PERMISSIONS = {
    # Admin & Team Management
    "create_admin": ["super_admin"],
    "create_team_member": ["super_admin", "admin"],
    "manage_team": ["super_admin"],
    "manage_users": ["super_admin"],

    # Issue Actions
    "assign_issue": ["super_admin", "admin"],
    "view_all_issues": ["super_admin", "admin"],
    "view_assigned_issues": ["super_admin", "admin", "team_member"],
    "approve_all": ["super_admin", "admin"],
    "approve_assigned": ["super_admin", "admin", "team_member"],
    "decline_all": ["super_admin", "admin"],
    "decline_assigned": ["super_admin", "admin", "team_member"],
    "send_to_authority": ["super_admin", "admin"],
    "edit_report": ["super_admin", "admin"],

    # Pages / Sections
    "view_dashboard": ["super_admin", "admin", "team_member", "viewer"],
    "view_warroom": ["super_admin", "admin"],
    "view_reviews": ["super_admin", "admin", "team_member"],
    "view_users": ["super_admin"],
    "view_team": ["super_admin", "admin"],
    "view_analytics": ["super_admin", "admin", "viewer"],
    "view_audit": ["super_admin", "admin"],
    "view_mapping": ["super_admin", "admin"],
    "view_authorities": ["super_admin", "admin"],
    "view_settings": ["super_admin", "admin", "team_member", "viewer"],

    # Settings Tabs
    "settings_profile": ["super_admin", "admin", "team_member", "viewer"],
    "settings_notifications": ["super_admin", "admin", "team_member"],
    "settings_security": ["super_admin", "admin", "team_member"],
    "settings_system": ["super_admin"],

    # System
    "change_system_settings": ["super_admin"],
    "maintenance_mode": ["super_admin"],
    "change_password": ["super_admin", "admin", "team_member", "viewer"],
    "enable_2fa": ["super_admin", "admin", "team_member"],
    
    # Legacy
    "view_stats": ["super_admin", "admin", "viewer"],
}

def has_permission(user_role: str, permission: str) -> bool:
    if not user_role or not permission:
        return False
    allowed_roles = PERMISSIONS.get(permission, [])
    return user_role in allowed_roles
