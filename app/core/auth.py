from fastapi import HTTPException, Depends, Header, status
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import logging
from utils.security import SECRET_KEY, ALGORITHM
from services.mongodb_optimized_service import get_optimized_mongodb_service

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
        if email is None:
            raise credentials_exception
        
        # Check if user exists in DB and is active
        # Optimization: We could trust the token until expiry, but checking DB allows revocation
        mongo_service = await get_optimized_mongodb_service()
        if mongo_service:
            try:
                collection = await mongo_service.get_collection("admins")
                user = await collection.find_one({"email": email})
                
                if user:
                    if not user.get("is_active", True):
                        raise HTTPException(status_code=403, detail="Inactive user")
                    
                    return {
                        "id": str(user["_id"]),
                        "email": user["email"],
                        "role": user.get("role", "admin"),
                        "name": user.get("name", "Admin")
                    }
            except Exception as e:
                logger.warning(f"Auth DB check failed, falling back to token validation: {e}")
            
        # Fallback if DB check fails but token is valid (should ideally not happen)
        return {"email": email, "role": "admin"}
        
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