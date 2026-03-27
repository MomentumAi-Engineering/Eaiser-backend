from fastapi import HTTPException, Depends, Header, status
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import logging
from utils.security import SECRET_KEY, ALGORITHM
from services.mongodb_optimized_service import get_optimized_mongodb_service
from core.permissions import has_permission, PERMISSIONS

logger = logging.getLogger(__name__)

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Validates JWT token from Authorization header.
    Returns the user payload if valid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not authorization or not authorization.startswith("Bearer "):
        # Check for demo token fallback for dev continuity (optional, can be removed for strict mode)
        # if authorization and "demo_token" in authorization: return {"role": "admin", "id": "demo"}
        raise credentials_exception

    token = authorization.split(" ")[1]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_type = payload.get("type", "admin") # default to admin token
        
        if email is None:
            raise credentials_exception
            
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            return {"email": email, "role": "admin"} # Fallback if DB is down during startup

        # Select appropriate collection based on token type
        collection_name = "government_users" if token_type == "gov_portal" else "admins"
        collection = await mongo_service.get_collection(collection_name)
        user = await collection.find_one({"email": email})
        
        if not user:
            raise credentials_exception
            
        if not user.get("is_active", True):
            raise HTTPException(status_code=403, detail="Account is deactivated")
            
        # Return unified user object
        user_data = {
            "id": str(user["_id"]),
            "email": user["email"],
            "role": user.get("role", "admin"),
            "name": user.get("name", "User"),
            "type": token_type
        }

        # Add extra context for gov officials
        if token_type == "gov_portal":
             user_data["dept"] = user.get("department")
             user_data["zip"] = user.get("zip_code")
             user_data["org"] = user.get("city")

        return user_data
        
    except JWTError:
        raise credentials_exception

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Admin-only authentication dependency - allows all admin roles
    """
    # Allow all admin roles: super_admin, admin, team_member, viewer
    valid_roles = ["admin", "super_admin", "team_member", "viewer"]
    
    if current_user.get("role") not in valid_roles:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user

def require_permission(permission: str):
    """
    Dependency factory to check for specific permission
    """
    async def permission_checker(current_user: Dict[str, Any] = Depends(get_admin_user)):
        if not has_permission(current_user.get("role"), permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required for this action"
            )
        return current_user
    return permission_checker