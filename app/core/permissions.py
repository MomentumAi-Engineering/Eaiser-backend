"""
Role-Based Access Control Middleware
"""
from fastapi import HTTPException, Depends
from typing import List, Callable
from core.auth import get_admin_user
import logging

logger = logging.getLogger(__name__)

class PermissionChecker:
    """
    Permission checker based on route access matrix
    """
    
    # Route Access Matrix
    PERMISSIONS = {
        "create_admin": ["super_admin"],
        "create_team_member": ["super_admin", "admin"],
        "assign_issue": ["super_admin", "admin"],
        "view_all_issues": ["super_admin", "admin"],
        "view_assigned_issues": ["super_admin", "admin", "team_member"],
        "approve_assigned": ["super_admin", "admin", "team_member"],
        "decline_assigned": ["super_admin", "admin", "team_member"],
        "view_stats": ["super_admin", "admin", "viewer"],
        "manage_team": ["super_admin"]
    }
    
    @staticmethod
    def check_permission(required_permission: str):
        """
        Decorator to check if admin has required permission
        """
        def permission_decorator(func: Callable):
            async def wrapper(*args, current_admin: dict = None, **kwargs):
                if current_admin is None:
                    # Try to get from kwargs
                    current_admin = kwargs.get('current_admin')
                
                if not current_admin:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )
                
                admin_role = current_admin.get("role", "viewer")
                allowed_roles = PermissionChecker.PERMISSIONS.get(required_permission, [])
                
                if admin_role not in allowed_roles:
                    logger.warning(
                        f"Permission denied: {current_admin.get('email')} "
                        f"(role: {admin_role}) attempted {required_permission}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
                    )
                
                logger.info(
                    f"✅ Permission granted: {current_admin.get('email')} "
                    f"-> {required_permission}"
                )
                
                return await func(*args, current_admin=current_admin, **kwargs)
            
            return wrapper
        return permission_decorator
    
    @staticmethod
    def has_permission(admin_role: str, permission: str) -> bool:
        """
        Check if a role has a specific permission
        """
        allowed_roles = PermissionChecker.PERMISSIONS.get(permission, [])
        return admin_role in allowed_roles


def require_permission(permission: str):
    """
    Dependency to require specific permission
    """
    async def permission_dependency(current_admin: dict = Depends(get_admin_user)):
        admin_role = current_admin.get("role", "viewer")
        
        if not PermissionChecker.has_permission(admin_role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_admin
    
    return permission_dependency


def require_role(allowed_roles: List[str]):
    """
    Dependency to require specific role(s)
    """
    async def role_dependency(current_admin: dict = Depends(get_admin_user)):
        admin_role = current_admin.get("role", "viewer")
        
        if admin_role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
            )
        
        return current_admin
    
    return role_dependency
