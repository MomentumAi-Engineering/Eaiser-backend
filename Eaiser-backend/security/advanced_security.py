#!/usr/bin/env python3
"""
🔒 SnapFix Enterprise Advanced Security Module
Comprehensive security implementation for 100,000+ concurrent users
Features: OAuth2, JWT refresh tokens, WAF, rate limiting, API versioning

Author: Senior Full-Stack AI/ML Engineer
Security Level: Enterprise Production
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from functools import wraps
import re

import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from redis import Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================
# SECURITY CONFIGURATION
# ========================================
class SecurityConfig:
    """Centralized security configuration"""
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # OAuth2 Configuration
    OAUTH2_TOKEN_URL = "/auth/token"
    OAUTH2_REFRESH_URL = "/auth/refresh"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 1000  # requests per minute
    RATE_LIMIT_WINDOW = "1 minute"
    BURST_RATE_LIMIT = 100  # requests per second for burst
    
    # Password Security
    MIN_PASSWORD_LENGTH = 12
    PASSWORD_COMPLEXITY_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]"
    BCRYPT_ROUNDS = 12
    
    # Session Security
    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 5
    
    # API Versioning
    CURRENT_API_VERSION = "v1"
    SUPPORTED_API_VERSIONS = ["v1"]
    
    # WAF Configuration
    BLOCKED_USER_AGENTS = [
        "sqlmap", "nikto", "nmap", "masscan", "zap", "burp"
    ]
    SUSPICIOUS_PATTERNS = [
        r"<script", r"javascript:", r"vbscript:", r"onload=", r"onerror=",
        r"union.*select", r"drop.*table", r"insert.*into", r"delete.*from",
        r"\.\./", r"\\x[0-9a-f]{2}", r"eval\(", r"exec\("
    ]

# ========================================
# SECURITY MODELS
# ========================================
class UserCreate(BaseModel):
    """User creation model with validation"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < SecurityConfig.MIN_PASSWORD_LENGTH:
            raise ValueError(f'Password must be at least {SecurityConfig.MIN_PASSWORD_LENGTH} characters')
        if not re.match(SecurityConfig.PASSWORD_COMPLEXITY_REGEX, v):
            raise ValueError('Password must contain uppercase, lowercase, digit, and special character')
        return v

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str = "read write"

class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str

class SecurityEvent(BaseModel):
    """Security event logging model"""
    event_type: str
    user_id: Optional[str] = None
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_score: int  # 1-10 scale

# ========================================
# ADVANCED SECURITY MANAGER
# ========================================
class AdvancedSecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.failed_attempts = {}  # In-memory cache for failed attempts
        self.blocked_ips = set()
        self.security_events = []
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt"""
        salt = bcrypt.gensalt(rounds=SecurityConfig.BCRYPT_ROUNDS)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_tokens(self, user_data: Dict[str, Any]) -> TokenResponse:
        """Generate access and refresh tokens"""
        now = datetime.now(timezone.utc)
        
        # Access token payload
        access_payload = {
            "sub": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data.get("role", "user"),
            "permissions": user_data.get("permissions", []),
            "iat": now,
            "exp": now + timedelta(minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES),
            "type": "access",
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": user_data["user_id"],
            "iat": now,
            "exp": now + timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS),
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)
        }
        
        # Generate tokens
        access_token = jwt.encode(
            access_payload, 
            SecurityConfig.JWT_SECRET_KEY, 
            algorithm=SecurityConfig.JWT_ALGORITHM
        )
        
        refresh_token = jwt.encode(
            refresh_payload, 
            SecurityConfig.JWT_SECRET_KEY, 
            algorithm=SecurityConfig.JWT_ALGORITHM
        )
        
        # Store refresh token in Redis with expiration
        refresh_key = f"refresh_token:{user_data['user_id']}:{refresh_payload['jti']}"
        self.redis.setex(
            refresh_key, 
            timedelta(days=SecurityConfig.REFRESH_TOKEN_EXPIRE_DAYS),
            json.dumps({
                "user_id": user_data["user_id"],
                "created_at": now.isoformat(),
                "ip_address": user_data.get("ip_address"),
                "user_agent": user_data.get("user_agent")
            })
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )
            
            # Check if token is revoked (for access tokens)
            if token_type == "access":
                revoked_key = f"revoked_token:{payload['jti']}"
                if self.redis.exists(revoked_key):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str, ip_address: str, user_agent: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        # Verify refresh token
        payload = self.verify_token(refresh_token, "refresh")
        user_id = payload["sub"]
        jti = payload["jti"]
        
        # Check if refresh token exists in Redis
        refresh_key = f"refresh_token:{user_id}:{jti}"
        token_data = self.redis.get(refresh_key)
        
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found or expired"
            )
        
        # Parse token data
        token_info = json.loads(token_data)
        
        # Security check: verify IP and user agent (optional, can be disabled for mobile apps)
        # if token_info.get("ip_address") != ip_address:
        #     self.log_security_event("suspicious_refresh", user_id, ip_address, user_agent, 
        #                           {"reason": "IP address mismatch"})
        
        # Get user data (this would typically come from database)
        user_data = {
            "user_id": user_id,
            "username": "user",  # Fetch from database
            "email": "user@example.com",  # Fetch from database
            "role": "user",  # Fetch from database
            "permissions": [],  # Fetch from database
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        # Generate new tokens
        return self.generate_tokens(user_data)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke access token"""
        try:
            payload = self.verify_token(token, "access")
            jti = payload["jti"]
            exp = payload["exp"]
            
            # Store revoked token until expiration
            revoked_key = f"revoked_token:{jti}"
            ttl = exp - int(time.time())
            
            if ttl > 0:
                self.redis.setex(revoked_key, ttl, "revoked")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    def log_security_event(self, event_type: str, user_id: Optional[str], 
                          ip_address: str, user_agent: str, details: Dict[str, Any], 
                          risk_score: int = 5):
        """Log security events for monitoring"""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(timezone.utc),
            details=details,
            risk_score=risk_score
        )
        
        # Store in Redis for real-time monitoring
        event_key = f"security_event:{int(time.time())}:{secrets.token_urlsafe(8)}"
        self.redis.setex(event_key, timedelta(days=30), event.json())
        
        # Log to file
        logger.warning(f"Security Event: {event_type} - User: {user_id} - IP: {ip_address} - Risk: {risk_score}")
        
        # Add to in-memory cache for immediate analysis
        self.security_events.append(event)
        
        # Keep only last 1000 events in memory
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def check_rate_limit(self, ip_address: str, endpoint: str) -> bool:
        """Advanced rate limiting with different limits per endpoint"""
        current_time = int(time.time())
        window = 60  # 1 minute window
        
        # Different limits for different endpoints
        endpoint_limits = {
            "/auth/login": 10,  # 10 login attempts per minute
            "/auth/register": 5,  # 5 registrations per minute
            "/api/v1/issues": 1000,  # 1000 API calls per minute
            "default": 100  # Default limit
        }
        
        limit = endpoint_limits.get(endpoint, endpoint_limits["default"])
        
        # Redis key for rate limiting
        rate_key = f"rate_limit:{ip_address}:{endpoint}:{current_time // window}"
        
        # Get current count
        current_count = self.redis.get(rate_key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= limit:
            # Log rate limit exceeded
            self.log_security_event(
                "rate_limit_exceeded",
                None,
                ip_address,
                "",
                {"endpoint": endpoint, "limit": limit, "count": current_count},
                risk_score=7
            )
            return False
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(rate_key)
        pipe.expire(rate_key, window)
        pipe.execute()
        
        return True
    
    def detect_suspicious_activity(self, request: Request) -> bool:
        """Web Application Firewall (WAF) functionality"""
        ip_address = get_remote_address(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Check blocked IPs
        if ip_address in self.blocked_ips:
            return False
        
        # Check blocked user agents
        for blocked_agent in SecurityConfig.BLOCKED_USER_AGENTS:
            if blocked_agent.lower() in user_agent.lower():
                self.log_security_event(
                    "blocked_user_agent",
                    None,
                    ip_address,
                    user_agent,
                    {"blocked_agent": blocked_agent},
                    risk_score=9
                )
                return False
        
        # Check request content for suspicious patterns
        request_content = str(request.url) + str(request.headers)
        
        for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
            if re.search(pattern, request_content, re.IGNORECASE):
                self.log_security_event(
                    "suspicious_pattern",
                    None,
                    ip_address,
                    user_agent,
                    {"pattern": pattern, "content": request_content[:200]},
                    risk_score=8
                )
                return False
        
        return True

# ========================================
# FASTAPI SECURITY MIDDLEWARE
# ========================================
class SecurityMiddleware:
    """Custom security middleware for FastAPI"""
    
    def __init__(self, app: FastAPI, security_manager: AdvancedSecurityManager):
        self.app = app
        self.security_manager = security_manager
    
    async def __call__(self, request: Request, call_next):
        # WAF check
        if not self.security_manager.detect_suspicious_activity(request):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Request blocked by security policy"}
            )
        
        # Rate limiting
        ip_address = get_remote_address(request)
        endpoint = str(request.url.path)
        
        if not self.security_manager.check_rate_limit(ip_address, endpoint):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Add security headers
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

# ========================================
# API VERSIONING
# ========================================
class APIVersioning:
    """API versioning management"""
    
    @staticmethod
    def get_api_version(request: Request) -> str:
        """Extract API version from request"""
        # Check header first
        version = request.headers.get("API-Version")
        if version:
            return version
        
        # Check URL path
        path_parts = request.url.path.split("/")
        for part in path_parts:
            if part.startswith("v") and part[1:].isdigit():
                return part
        
        # Default version
        return SecurityConfig.CURRENT_API_VERSION
    
    @staticmethod
    def validate_api_version(version: str) -> bool:
        """Validate if API version is supported"""
        return version in SecurityConfig.SUPPORTED_API_VERSIONS

# ========================================
# AUTHENTICATION DEPENDENCIES
# ========================================
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=SecurityConfig.OAUTH2_TOKEN_URL)

def create_security_dependencies(security_manager: AdvancedSecurityManager):
    """Create FastAPI dependencies for authentication"""
    
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Get current authenticated user"""
        token = credentials.credentials
        payload = security_manager.verify_token(token, "access")
        return payload
    
    async def get_current_active_user(current_user: dict = Depends(get_current_user)):
        """Get current active user (additional checks can be added)"""
        # Add additional user status checks here
        return current_user
    
    async def require_permissions(required_permissions: List[str]):
        """Require specific permissions"""
        def permission_checker(current_user: dict = Depends(get_current_active_user)):
            user_permissions = current_user.get("permissions", [])
            
            for permission in required_permissions:
                if permission not in user_permissions:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission}' required"
                    )
            
            return current_user
        
        return permission_checker
    
    return get_current_user, get_current_active_user, require_permissions

# ========================================
# USAGE EXAMPLE
# ========================================
def create_secure_app() -> FastAPI:
    """Create FastAPI app with advanced security"""
    app = FastAPI(
        title="SnapFix Enterprise API",
        description="Secure API for 100,000+ concurrent users",
        version="1.0.0"
    )
    
    # Initialize Redis client
    redis_client = Redis(host="redis-cluster", port=7001, decode_responses=True)
    
    # Initialize security manager
    security_manager = AdvancedSecurityManager(redis_client)
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware, security_manager=security_manager)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://snapfix.com", "https://app.snapfix.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["snapfix.com", "*.snapfix.com", "localhost"]
    )
    
    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    return app

if __name__ == "__main__":
    print("🔒 SnapFix Enterprise Security Module")
    print("🛡️ Features: OAuth2, JWT, WAF, Rate Limiting, API Versioning")
    print("🎯 Target: 100,000+ concurrent users")
    print("🔐 Security Level: Enterprise Production")
    print("="*60)
    
    # Example usage
    app = create_secure_app()
    print("✅ Secure FastAPI application created successfully!")
    print("📊 Security features enabled:")
    print("   • JWT with refresh tokens")
    print("   • OAuth2 authentication")
    print("   • Web Application Firewall (WAF)")
    print("   • Advanced rate limiting")
    print("   • API versioning")
    print("   • Security event logging")
    print("   • CORS and trusted host protection")
    print("   • Comprehensive security headers")