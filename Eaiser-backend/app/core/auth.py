# 🔐 Authentication Module for API Security
# Simple authentication system for API endpoints

from fastapi import HTTPException, Depends, Header
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Simple authentication dependency for API endpoints
    In production, this would validate JWT tokens or API keys
    """
    
    # For demo purposes, return a mock user
    # In production, validate the authorization header
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        # Mock token validation - in production, validate JWT
        if token == "demo_token":
            return {
                "user_id": "demo_user",
                "username": "demo",
                "role": "admin",
                "permissions": ["read", "write", "admin"]
            }
    
    # For development, allow requests without authentication
    return {
        "user_id": "anonymous",
        "username": "anonymous",
        "role": "user",
        "permissions": ["read"]
    }

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Admin-only authentication dependency
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user

def create_demo_token() -> str:
    """
    Create a demo token for testing
    """
    return "demo_token"