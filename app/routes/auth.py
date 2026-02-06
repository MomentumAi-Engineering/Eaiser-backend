from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from services.mongodb_service import get_db
from utils.security import verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM
from google.oauth2 import id_token
from google.auth.transport import requests
import os
import logging

# Setup Logging
logger = logging.getLogger(__name__)

router = APIRouter()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
logger.info(f"Google Auth Configured with Client ID: {GOOGLE_CLIENT_ID[:10]}..." if GOOGLE_CLIENT_ID else "GOOGLE_CLIENT_ID NOT SET")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Schema definitions
class UserCreate(BaseModel):
    name: str = "User"
    fullName: Optional[str] = None # Added for compatibility with new frontend
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleLogin(BaseModel):
    credential: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ... (intervening lines)

@router.post("/signup", response_model=Token)
async def signup(user: UserCreate):
    try:
        db = await get_db()
        # Check if user exists
        existing_user = await db["users"].find_one({"email": user.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash password and store
        hashed_password = get_password_hash(user.password)
        
        # Handle logic for fullName/name compatibility
        final_name = user.name
        if user.fullName and (user.name == "User" or not user.name):
            final_name = user.fullName
            
        new_user = {
            "name": final_name,
            "email": user.email,
            "hashed_password": hashed_password,
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        result = await db["users"].insert_one(new_user)
        new_user["_id"] = str(result.inserted_id)
        
        # Create token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": "user", "id": str(result.inserted_id)},
            expires_delta=access_token_expires
        )
        
        # Send Welcome Email
        try:
            from services.email_service import send_user_welcome_email
            await send_user_welcome_email(user.email, user.name)
        except Exception as email_error:
             # Log but don't fail the signup
            logger.error(f"Failed to send welcome email: {email_error}")

        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": str(result.inserted_id),
                "name": user.name,
                "email": user.email,
                "role": "user"
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Signup error: {e}\nTraceback:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    try:
        db = await get_db()
        user = await db["users"].find_one({"email": user_data.email})
        
        # Check if user exists AND has a password (google-auth users might not have one)
        if not user or "hashed_password" not in user or not verify_password(user_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.get("is_active", True):
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Please contact support."
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"], "role": user.get("role", "user"), "id": str(user["_id"])},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": str(user["_id"]),
                "name": user.get("name", "User"),
                "email": user.get("email"),
                "role": user.get("role", "user")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/google", response_model=Token)
async def google_login(login_data: GoogleLogin):
    try:
        # Debug incoming credential
        logger.info(f"Received Google Login Request. Credential prefix: {login_data.credential[:10]}...")
        if not GOOGLE_CLIENT_ID:
            logger.error("GOOGLE_CLIENT_ID is missing in server environment.")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing Google Client ID")

        try:
             id_info = id_token.verify_oauth2_token(login_data.credential, requests.Request(), GOOGLE_CLIENT_ID)
        except ValueError as ve:
             logger.error(f"Google Token Verification ValueError: {ve}")
             # Detailed error for debugging (remove in prod if needed, but useful now)
             raise HTTPException(status_code=400, detail=f"Invalid Token: {str(ve)}")
        
        if not id_info:
            raise HTTPException(status_code=400, detail="Invalid Google Token")

        email = id_info.get("email")
        name = id_info.get("name")
        
        db = await get_db()
        user = await db["users"].find_one({"email": email})
        
        if not user:
            # Create user if logging in for first time via Google
            user_payload = {
                "name": name,
                "email": email,
                "role": "user",
                "is_active": True,
                "auth_provider": "google",
                "created_at": datetime.utcnow()
            }
            result = await db["users"].insert_one(user_payload)
            user_id = str(result.inserted_id)
            user = {**user_payload, "_id": user_id} # Construct user object for response
            
            # Send Welcome Email for new Google User
            try:
                from services.email_service import send_user_welcome_email
                await send_user_welcome_email(email, name)
            except Exception as email_error:
                logger.error(f"Failed to send welcome email to Google user: {email_error}")

        else:
            if not user.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated. Please contact support."
                )
            user_id = str(user["_id"])

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email, "role": user.get("role", "user"), "id": user_id},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "name": user.get("name"),
                "email": user.get("email"),
                "role": user.get("role", "user")
            }
        }

    except ValueError as e:
         logger.error(f"Google Token Verification Failed: {e}")
         raise HTTPException(status_code=400, detail="Invalid Google Token")
    except Exception as e:
        logger.error(f"Google Login Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
