
# Role-Based Access Control Matrix
# Matches frontend src/utils/permissions.js

PERMISSIONS = {
    # Admin & Team Management
    "create_admin": ["super_admin"],
    "create_team_member": ["super_admin", "admin"],
    "manage_team": ["super_admin"],
    "manage_users": ["super_admin"],

    # Issue Actions
    "assign_issue": ["super_admin", "admin", "operations", "ops_manager", "department_admin"],
    # Staff (and above) verify/decline a crew member's resolution photo before a
    # report can be marked resolved. Crew CANNOT self-verify.
    "verify_resolution": ["super_admin", "admin", "operations", "ops_manager", "department_admin"],
    "view_all_issues": ["super_admin", "admin"],
    "view_assigned_issues": ["super_admin", "admin", "team_member"],
    "approve_all": ["super_admin", "admin"],
    "approve_assigned": ["super_admin", "admin", "team_member"],
    "decline_all": ["super_admin", "admin"],
    "decline_assigned": ["super_admin", "admin", "team_member"],
    "send_to_authority": ["super_admin", "admin"],
    "edit_report": ["super_admin", "admin"],

    # Mayor / Council — READ-ONLY executive view. They can see dashboards,
    # analytics, stats and resolved evidence, but cannot manage or configure
    # anything (no assign/verify/manage_authorities/routing config).
    #
    # Onboarding Specialist — READ-ONLY setup reviewer. Guides the City Manager
    # through setup and reviews the routing config / Tier-0 safety floors before
    # go-live, but the City Manager OWNS the config (specialist cannot edit it).
    # Pages / Sections
    "view_dashboard": ["super_admin", "admin", "team_member", "viewer", "mayor", "onboarding_specialist"],
    "view_warroom": ["super_admin", "admin"],
    "view_reviews": ["super_admin", "admin", "team_member"],
    "view_users": ["super_admin"],
    "view_team": ["super_admin", "admin"],
    "view_analytics": ["super_admin", "admin", "viewer", "mayor", "onboarding_specialist"],
    "view_audit": ["super_admin", "admin"],
    "view_mapping": ["super_admin", "admin"],
    "view_authorities": ["super_admin", "admin", "operations", "ops_manager", "department_admin", "onboarding_specialist"],
    "manage_authorities": ["super_admin", "admin", "ops_manager", "operations", "department_admin"],
    "view_settings": ["super_admin", "admin", "team_member", "viewer", "onboarding_specialist"],

    # Settings Tabs
    "settings_profile": ["super_admin", "admin", "team_member", "viewer"],
    "settings_notifications": ["super_admin", "admin", "team_member"],
    "settings_security": ["super_admin", "admin", "team_member"],
    "settings_system": ["super_admin"],

    # Routing Config Module — City Manager (Super Admin) owns/edits it; the
    # Onboarding Specialist may VIEW it (review Tier-0 floors before go-live).
    "view_routing_config": ["super_admin", "admin", "onboarding_specialist"],
    "manage_routing_config": ["super_admin", "admin"],

    # Create a brand-new city tenant — IT Super Admin (SYSTEM) only. A city-level
    # Super Admin can NOT create other cities; that's a platform/system action.
    "create_city_tenant": ["it_super_admin"],

    # System
    "change_system_settings": ["super_admin"],
    "maintenance_mode": ["super_admin"],
    "change_password": ["super_admin", "admin", "team_member", "viewer", "operations", "ops_manager", "department_admin", "onboarding_specialist"],
    "enable_2fa": ["super_admin", "admin", "team_member"],

    # Subscriptions & Billing (mirror frontend)
    "view_subscriptions": ["super_admin", "admin"],
    "manage_subscriptions": ["super_admin"],
    
    # Legacy
    "view_stats": ["super_admin", "admin", "viewer", "mayor", "onboarding_specialist"],
}

def has_permission(user_role: str, permission: str) -> bool:
    if not user_role or not permission:
        return False
    allowed_roles = PERMISSIONS.get(permission, [])
    if user_role in allowed_roles:
        return True
    # IT Super Admin (SYSTEM) inherits EVERY City Super Admin permission, plus the
    # system-only ones (create_city_tenant) it's listed on directly above. So we
    # only ever maintain super_admin in the matrix and IT inherits it here.
    if user_role == "it_super_admin" and "super_admin" in allowed_roles:
        return True
    return False
