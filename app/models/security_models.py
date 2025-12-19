"""
Enhanced Admin Security Models
"""
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
import re

class LoginAttempt(BaseModel):
    """Track login attempts for security"""
    email: str
    ip_address: str
    user_agent: str
    success: bool
    timestamp: datetime
    failure_reason: Optional[str] = None

class AdminSession(BaseModel):
    """Active admin session tracking"""
    admin_email: str
    session_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_active: bool = True

class PasswordChangeRequest(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Enforce strong password requirements"""
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        # Check for common passwords
        common_passwords = ['password', '12345678', 'qwerty', 'admin123']
        if v.lower() in common_passwords:
            raise ValueError('Password is too common. Please choose a stronger password')
        
        return v

class TwoFactorSetup(BaseModel):
    """2FA setup model"""
    email: str
    method: str  # 'email' or 'totp'
    
class TwoFactorVerify(BaseModel):
    """2FA verification model"""
    email: str
    code: str
    session_token: str

class AdminSecuritySettings(BaseModel):
    """Admin security preferences"""
    email: str
    two_factor_enabled: bool = False
    two_factor_method: Optional[str] = None  # 'email' or 'totp'
    ip_whitelist: List[str] = []
    require_password_change: bool = False
    password_last_changed: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None
    last_login_ip: Optional[str] = None

class SecurityAuditLog(BaseModel):
    """Security audit trail"""
    admin_email: str
    action: str  # 'login', 'logout', 'password_change', 'role_change', etc.
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    details: Optional[dict] = None
