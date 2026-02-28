from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import jwt
import os

import bcrypt

# Password Hashing
# pwd_context removed in favor of direct bcrypt

import logging
logger = logging.getLogger(__name__)

# JWT Configuration
# In production, these should be loaded from env vars with secure defaults
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-should-be-in-env-file-and-very-secure")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY", SECRET_KEY + "-refresh-token-secret")
logger.info(f"🔰 Security Config: ALGORITHM=HS256, SECRET_KEY={'SET (' + SECRET_KEY[:4] + '...)' if SECRET_KEY else 'NOT SET'}")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days (extended for development convenience)
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days for refresh tokens

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # bcrypt requires bytes
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    if isinstance(password, str):
        password = password.encode('utf-8')
    # gensalt() generates a salt, hashpw hashes it
    return bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')

def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(subject: Union[str, Any], extra_claims: Optional[dict] = None) -> str:
    """Create a long-lived refresh token with separate secret."""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    if extra_claims:
        to_encode.update(extra_claims)
    encoded_jwt = jwt.encode(to_encode, REFRESH_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_refresh_token(token: str) -> Optional[dict]:
    """Verify and decode a refresh token. Returns payload or None."""
    try:
        payload = jwt.decode(token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except Exception:
        return None
