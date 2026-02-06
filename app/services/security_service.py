"""
Advanced Security Service
Handles rate limiting, account lockout, 2FA, and security logging
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
import secrets
import pyotp
from fastapi import HTTPException, Request
from services.mongodb_optimized_service import get_optimized_mongodb_service
from models.security_models import LoginAttempt, SecurityAuditLog, AdminSecuritySettings

logger = logging.getLogger(__name__)

class SecurityService:
    """Advanced security features for admin authentication"""
    
    # Configuration
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    RATE_LIMIT_WINDOW_SECONDS = 60
    MAX_REQUESTS_PER_WINDOW = 10
    
    @staticmethod
    async def check_rate_limit(ip_address: str) -> bool:
        """
        Check if IP has exceeded rate limit
        Returns True if allowed, False if rate limited
        """
        try:
            mongo_service = await get_optimized_mongodb_service()
            if not mongo_service:
                return True  # Fail open if service unavailable
            
            attempts_collection = await mongo_service.get_collection("login_attempts")
            
            # Count attempts in last window
            window_start = datetime.utcnow() - timedelta(seconds=SecurityService.RATE_LIMIT_WINDOW_SECONDS)
            
            recent_attempts = await attempts_collection.count_documents({
                "ip_address": ip_address,
                "timestamp": {"$gte": window_start}
            })
            
            if recent_attempts >= SecurityService.MAX_REQUESTS_PER_WINDOW:
                logger.warning(f"🚫 Rate limit exceeded for IP: {ip_address}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Fail open
    
    @staticmethod
    async def check_account_lockout(email: str) -> Dict:
        """
        Check if account is locked due to failed login attempts
        Returns dict with 'locked' status and 'unlock_time'
        """
        try:
            mongo_service = await get_optimized_mongodb_service()
            if not mongo_service:
                return {"locked": False}
            
            admins_collection = await mongo_service.get_collection("admins")
            admin = await admins_collection.find_one({"email": email})
            
            if not admin:
                return {"locked": False}
            
            # Check if account is locked
            locked_until = admin.get("account_locked_until")
            if locked_until and locked_until > datetime.utcnow():
                remaining = (locked_until - datetime.utcnow()).total_seconds() / 60
                logger.warning(f"🔒 Account locked for {email}, {remaining:.1f} minutes remaining")
                return {
                    "locked": True,
                    "unlock_time": locked_until,
                    "remaining_minutes": int(remaining)
                }
            
            # Clear lockout if expired
            if locked_until and locked_until <= datetime.utcnow():
                await admins_collection.update_one(
                    {"email": email},
                    {
                        "$set": {"account_locked_until": None, "failed_login_attempts": 0}
                    }
                )
            
            return {"locked": False}
            
        except Exception as e:
            logger.error(f"Error checking account lockout: {e}")
            return {"locked": False}  # Fail open
    
    @staticmethod
    async def record_login_attempt(
        email: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        failure_reason: Optional[str] = None
    ):
        """Record login attempt for security tracking"""
        try:
            mongo_service = await get_optimized_mongodb_service()
            if not mongo_service:
                return
            
            # Record attempt
            attempts_collection = await mongo_service.get_collection("login_attempts")
            attempt = LoginAttempt(
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                timestamp=datetime.utcnow(),
                failure_reason=failure_reason
            )
            await attempts_collection.insert_one(attempt.dict())
            
            # Update admin record
            admins_collection = await mongo_service.get_collection("admins")
            
            if success:
                # Reset failed attempts on successful login
                await admins_collection.update_one(
                    {"email": email},
                    {
                        "$set": {
                            "failed_login_attempts": 0,
                            "last_login": datetime.utcnow(),
                            "last_login_ip": ip_address,
                            "account_locked_until": None
                        }
                    }
                )
                logger.info(f"✅ Successful login for {email} from {ip_address}")
            else:
                # Increment failed attempts
                admin = await admins_collection.find_one({"email": email})
                if admin:
                    failed_attempts = admin.get("failed_login_attempts", 0) + 1
                    
                    update_data = {
                        "failed_login_attempts": failed_attempts
                    }
                    
                    # Lock account if max attempts exceeded
                    if failed_attempts >= SecurityService.MAX_LOGIN_ATTEMPTS:
                        lockout_until = datetime.utcnow() + timedelta(minutes=SecurityService.LOCKOUT_DURATION_MINUTES)
                        update_data["account_locked_until"] = lockout_until
                        logger.warning(f"🔒 Account locked for {email} until {lockout_until}")
                    
                    await admins_collection.update_one(
                        {"email": email},
                        {"$set": update_data}
                    )
                    
                logger.warning(f"❌ Failed login attempt for {email} from {ip_address}: {failure_reason}")
            
        except Exception as e:
            logger.error(f"Error recording login attempt: {e}")
    
    @staticmethod
    async def log_security_event(
        admin_email: str,
        action: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[dict] = None
    ):
        """Log security-related events for audit trail"""
        try:
            mongo_service = await get_optimized_mongodb_service()
            if not mongo_service:
                return
            
            # UNIFIED: Use 'audit_logs' for everything so frontend can display single timeline
            audit_collection = await mongo_service.get_collection("audit_logs")
            
            log_entry = SecurityAuditLog(
                admin_email=admin_email,
                action=action,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.utcnow(),
                success=success,
                details=details
            )
            
            # Use loose schema to accommodate different log types
            await audit_collection.insert_one(log_entry.dict())
            logger.info(f"📝 Security event logged: {action} by {admin_email}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    @staticmethod
    def generate_2fa_secret() -> str:
        """Generate TOTP secret for 2FA"""
        return pyotp.random_base32()
    
    @staticmethod
    def verify_2fa_code(secret: str, code: str) -> bool:
        """Verify TOTP code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)
    
    @staticmethod
    def generate_email_2fa_code() -> str:
        """Generate 6-digit code for email 2FA"""
        return str(secrets.randbelow(1000000)).zfill(6)
    
    @staticmethod
    async def send_2fa_email(email: str, code: str):
        """Send 2FA code via email"""
        try:
            from services.email_service import send_email
            
            subject = "EAiSER Admin - Two-Factor Authentication Code"
            html_content = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2>Two-Factor Authentication</h2>
                <p>Your verification code is:</p>
                <div style="background: #f0f0f0; padding: 20px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 5px;">
                    {code}
                </div>
                <p>This code will expire in 10 minutes.</p>
                <p style="color: #999; font-size: 12px;">If you didn't request this code, please ignore this email.</p>
            </div>
            """
            
            text_content = f"Your 2FA code is: {code}. This code will expire in 10 minutes."
            
            await send_email(email, subject, html_content, text_content)
            logger.info(f"📧 2FA code sent to {email}")
            
        except Exception as e:
            logger.error(f"Error sending 2FA email: {e}")
            raise
    
    @staticmethod
    def get_client_info(request: Request) -> Dict:
        """Extract client IP and user agent from request"""
        # Try to get real IP from headers (for proxy/load balancer scenarios)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip_address = forwarded_for.split(",")[0].strip()
        else:
            ip_address = request.client.host if request.client else "unknown"
        
        user_agent = request.headers.get("User-Agent", "unknown")
        
        return {
            "ip_address": ip_address,
            "user_agent": user_agent
        }
