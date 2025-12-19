from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import jwt
import os

import bcrypt

# Password Hashing
# pwd_context removed in favor of direct bcrypt

# JWT Configuration
# In production, these should be loaded from env vars with secure defaults
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-should-be-in-env-file-and-very-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours for admin convenience

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
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
