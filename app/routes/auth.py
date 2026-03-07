from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
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
import secrets
import hashlib
import random
import string

# Setup Logging
logger = logging.getLogger(__name__)

def generate_short_id():
    """Generate a customer-friendly 7-character alphanumeric ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))

router = APIRouter()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
logger.info(f"Google Auth Configured with Client ID: {GOOGLE_CLIENT_ID[:10]}..." if GOOGLE_CLIENT_ID else "GOOGLE_CLIENT_ID NOT SET")

from fastapi.responses import JSONResponse

@router.get("/mapkit-token")
async def get_mapkit_token():
    try:
        team_id = os.environ.get("APPLE_TEAM_ID")
        key_id = os.environ.get("APPLE_MAPKIT_KEY_ID")
        private_key = os.environ.get("APPLE_MAPKIT_PRIVATE_KEY")
        
        if not team_id or not key_id or not private_key:
            return JSONResponse({"token": None, "error": "Apple MapKit configuration missing"}, status_code=500)

        # Ensure correct formatting for multiline env vars
        private_key = private_key.replace("\\n", "\n") 

        current_time = int(datetime.utcnow().timestamp())
        claims = {
            "iss": team_id,
            "iat": current_time,
            "exp": current_time + 1800, # valid for 30 mins
        }
        
        token = jwt.encode(claims, private_key, algorithm="ES256", headers={"kid": key_id})
        return {"token": token}
    except Exception as e:
        logger.error(f"Error generating MapKit token: {str(e)}")
        return JSONResponse({"token": None, "error": str(e)}, status_code=500)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return {"sub": email, "id": payload.get("id"), "role": payload.get("role")}
    except JWTError:
        raise credentials_exception

class ProfileUpdate(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    username: Optional[str] = None

class PasswordChange(BaseModel):
    currentPassword: str
    newPassword: str

class NotificationUpdate(BaseModel):
    email: Optional[bool] = None
    push: Optional[bool] = None
    updates: Optional[bool] = None

# Schema definitions
class UserCreate(BaseModel):
    firstName: str
    lastName: str
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    identifier: str # Email or Username
    password: str

class GoogleLogin(BaseModel):
    credential: str

class AppleLogin(BaseModel):
    user: Optional[Any] = None # Can be dict or stringified JSON
    authorization: Dict[str, Any]

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

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

class VerifyEmailRequest(BaseModel):
    token: str

@router.post("/signup", response_model=Dict[str, Any])
async def signup(user: UserCreate, background_tasks: BackgroundTasks):
    try:
        db = await get_db()
        email = user.email.lower()
        username = user.username.lower()

        # Check if email exists
        if await db["users"].find_one({"email": email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Check if username exists
        if await db["users"].find_one({"username": username}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

        # Hash password
        hashed_password = get_password_hash(user.password)
        
        # Verification token
        verification_token = secrets.token_urlsafe(32)
            
        new_user = {
            "first_name": user.firstName,
            "last_name": user.lastName,
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
            "role": "user",
            "is_active": True,
            "email_verified": False,
            "verification_token": verification_token,
            "tos_accepted": True,
            "created_at": datetime.utcnow()
        }
        
        await db["users"].insert_one(new_user)
        
        # Send Verification Email in background
        try:
            from services.email_service import send_verification_email
            background_tasks.add_task(send_verification_email, email, user.firstName, verification_token)
        except Exception as email_error:
            logger.error(f"Failed to dispatch verification email: {email_error}")

        # Send TOS Email in background
        try:
            from services.email_service import send_tos_email
            background_tasks.add_task(send_tos_email, email, user.firstName)
        except Exception as tos_error:
            logger.error(f"Failed to dispatch TOS email for new user: {tos_error}")

        return {
            "message": "Account created successfully. Please check your email to verify your account.",
            "email": email
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
        identifier = user_data.identifier.lower()
        
        # Find by email or username
        user = await db["users"].find_one({
            "$or": [
                {"email": identifier},
                {"username": identifier}
            ]
        })
        
        if not user:
            logger.warning(f"❌ Login failed: User '{identifier}' not found.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect credentials", # Keep generic for security
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if Google account
        if "hashed_password" not in user:
             provider = user.get("auth_provider", "Google")
             logger.info(f"💡 User '{identifier}' found but uses {provider}. Guiding to Google Login.")
             raise HTTPException(
                 status_code=status.HTTP_401_UNAUTHORIZED,
                 detail=f"This account uses {provider.capitalize()} login. Please click 'Sign in with Google'.",
                 headers={"WWW-Authenticate": "Bearer"},
             )

        if not verify_password(user_data.password, user["hashed_password"]):
            logger.warning(f"❌ Login failed: Incorrect password for '{identifier}'.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.get("is_active", True):
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Please contact support."
            )

        # check email verification if needed
        # if not user.get("email_verified", False):
        #     raise HTTPException(status_code=403, detail="Please verify your email first.")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"], "role": user.get("role", "user"), "id": str(user["_id"])},
            expires_delta=access_token_expires
        )
        
        # Generate a display-friendly short ID
        raw_id = str(user["_id"])
        short_id = raw_id[-7:].upper()

        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": short_id,
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "username": user.get("username", ""),
                "email": user.get("email"),
                "role": user.get("role", "user"),
                "email_verified": user.get("email_verified", False)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/verify-email")
async def verify_email(request: VerifyEmailRequest):
    try:
        db = await get_db()
        user = await db["users"].find_one({"verification_token": request.token})
        
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired verification token")
            
        await db["users"].update_one(
            {"_id": user["_id"]},
            {
                "$set": {"email_verified": True},
                "$unset": {"verification_token": ""}
            }
        )
        
        # Optional: Send welcome email after verification if not sent before
        if not user.get("welcome_email_sent", False):
            try:
                from services.email_service import send_user_welcome_email
                await send_user_welcome_email(user["email"], user.get("first_name", "User"))
                await db["users"].update_one({"_id": user["_id"]}, {"$set": {"welcome_email_sent": True}})
            except:
                pass
            
        return {"message": "Email verified successfully. You can now log in."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/resend-verification")
async def resend_verification(current_user: dict = Depends(get_current_user)):
    """Resend verification email to the logged-in user."""
    try:
        db = await get_db()
        user = await db["users"].find_one({"email": current_user["sub"]})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.get("email_verified", False):
            return {"message": "Email is already verified."}
        
        # Generate new verification token
        new_token = secrets.token_urlsafe(32)
        await db["users"].update_one(
            {"_id": user["_id"]},
            {"$set": {"verification_token": new_token}}
        )
        
        from services.email_service import send_verification_email
        await send_verification_email(user["email"], user.get("first_name", "User"), new_token)
        
        logger.info(f"📧 Resent verification email to {user['email']}")
        return {"message": "Verification email sent successfully. Please check your inbox."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resend verification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to resend verification email")

@router.post("/google", response_model=Token)
async def google_login(login_data: GoogleLogin, background_tasks: BackgroundTasks):
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
        name = id_info.get("name", "")
        given_name = id_info.get("given_name", "")
        family_name = id_info.get("family_name", "")
        
        # Split name into first/last if Google didn't provide given_name
        if not given_name and name:
            parts = name.split(" ", 1)
            given_name = parts[0]
            family_name = parts[1] if len(parts) > 1 else ""
        
        # Generate a username from email
        base_username = email.split("@")[0].lower() if email else "user"
        
        db = await get_db()
        user = await db["users"].find_one({"email": email})
        
        if not user:
            # Create user if logging in for first time via Google
            user_payload = {
                "name": name,
                "first_name": given_name,
                "last_name": family_name,
                "username": base_username,
                "email": email,
                "role": "user",
                "is_active": True,
                "auth_provider": "google",
                "avatar": id_info.get("picture"),
                "email_verified": id_info.get("email_verified", False),
                "tos_accepted": True,
                "created_at": datetime.utcnow()
            }
            result = await db["users"].insert_one(user_payload)
            user_id = str(result.inserted_id)
            user = {**user_payload, "_id": user_id}

            # Send TOS email
            try:
                from services.email_service import send_tos_email
                background_tasks.add_task(send_tos_email, email, name)
            except Exception as tos_error:
                logger.error(f"Failed to send TOS email to Google user: {tos_error}")
        else:
            # For existing Google users: backfill missing first_name/last_name/username
            backfill = {}
            if not user.get("first_name") and given_name:
                backfill["first_name"] = given_name
            if not user.get("last_name") and family_name:
                backfill["last_name"] = family_name
            if not user.get("username") and base_username:
                backfill["username"] = base_username
            if not user.get("avatar") and id_info.get("picture"):
                backfill["avatar"] = id_info.get("picture")
            if backfill:
                await db["users"].update_one({"email": email}, {"$set": backfill})
                user.update(backfill)
            
        # Send Welcome Email for new Google User if not sent before
        if not user.get("welcome_email_sent", False):
            try:
                from services.email_service import send_user_welcome_email
                await send_user_welcome_email(email, name)
                await db["users"].update_one({"_id": user["_id"]}, {"$set": {"welcome_email_sent": True}})
            except Exception as email_error:
                logger.error(f"Failed to send welcome email to Google user: {email_error}")

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
        
        # Generate a display-friendly short ID
        raw_id = str(user["_id"])
        short_id = raw_id[-7:].upper()

        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": short_id,
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

@router.post("/apple", response_model=Token)
async def apple_login(login_data: AppleLogin, background_tasks: BackgroundTasks):
    try:
        # 1. Extensive Logging for Debugging
        logger.info(f"--- Apple Login Request ---")
        auth_info = login_data.authorization
        id_token_jwt = auth_info.get("id_token")
        
        # User info only comes on the very FIRST authorization by user
        # Apple sometimes sends this as a stringified JSON
        user_info = login_data.user
        if isinstance(user_info, str):
            try:
                import json
                user_info = json.loads(user_info)
                logger.info("Successfully parsed stringified user_info")
            except Exception as e:
                logger.error(f"Failed to parse stringified user_info: {e}")
                user_info = {}
        elif not user_info:
            user_info = {}
            
        logger.info(f"Final User Info: {user_info}")
        
        # 3. Extract Name from user_info (if provided)
        name_info = user_info.get("name", {})
        first_name = name_info.get("firstName", "")
        last_name = name_info.get("lastName", "")
        logger.info(f"Extracted Names: first='{first_name}', last='{last_name}'")

        # Decode id_token for email
        try:
            token_payload = jwt.get_unverified_claims(id_token_jwt)
            email = token_payload.get("email")
            if not email:
                raise ValueError("Apple token missing email")
        except Exception as e:
            logger.error(f"Failed to decode Apple id_token: {e}")
            raise HTTPException(status_code=400, detail="Invalid Apple Token")

        db = await get_db()
        email_lower = email.lower()
        user = await db["users"].find_one({"email": email_lower})

        base_username = email_lower.split("@")[0]
        
        if not user:
            # Create new user for first-time Apple authentication
            user_payload = {
                "first_name": first_name,
                "last_name": last_name,
                "username": base_username,
                "email": email_lower,
                "role": "user",
                "is_active": True,
                "auth_provider": "apple",
                "email_verified": True,
                "tos_accepted": True,
                "created_at": datetime.utcnow()
            }
            result = await db["users"].insert_one(user_payload)
            user_id = str(result.inserted_id)
            user = {**user_payload, "_id": user_id}
            logger.info(f"🆕 Registered new Apple User: {email_lower}")

            # Send TOS email
            try:
                from services.email_service import send_tos_email
                background_tasks.add_task(send_tos_email, email_lower, first_name)
            except Exception as tos_error:
                logger.error(f"Failed to send TOS email to Apple user: {tos_error}")
        else:
            user_id = str(user["_id"])
            logger.info(f"🔓 Existing Apple User found: {email_lower}")
            
            # 4. Backfill Logic: Update missing info if Apple provided it this time
            backfill = {}
            if not user.get("first_name") and first_name:
                backfill["first_name"] = first_name
            if not user.get("last_name") and last_name:
                backfill["last_name"] = last_name
            if user.get("auth_provider") != "apple":
                backfill["auth_provider"] = "apple"
            
            if backfill:
                logger.info(f"📝 Backfilling user data: {backfill}")
                await db["users"].update_one({"_id": user["_id"]}, {"$set": backfill})
                user.update(backfill)

        # Send Welcome Email for new Apple User if not sent before
        if not user.get("welcome_email_sent", False):
            try:
                from services.email_service import send_user_welcome_email
                await send_user_welcome_email(email_lower, user.get("first_name", ""))
                await db["users"].update_one({"_id": user["_id"]}, {"$set": {"welcome_email_sent": True}})
            except Exception as email_error:
                logger.error(f"Failed to send welcome email to Apple user: {email_error}")

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Please contact support."
            )

        # 5. Token Generation
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"], "role": user.get("role", "user"), "id": user_id},
            expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "username": user.get("username", ""),
                "email": user.get("email"),
                "role": user.get("role", "user")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple Login Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during Apple Auth")
@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    """
    Generate a secure reset token and send it via email.
    Always returns success to prevent email enumeration.
    """
    try:
        db = await get_db()
        email = request.email.lower()
        
        # 1. Check if user exists
        user = await db["users"].find_one({"email": email})
        
        if user:
            # 2. Generate secure token
            raw_token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            
            # 3. Store hashed token in DB with 15-min expiry
            await db["users"].update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "reset_password_token": token_hash,
                        "reset_password_expires": datetime.utcnow() + timedelta(minutes=15)
                    }
                }
            )
            
            # 4. Send email in background
            from services.email_service import send_password_reset_email
            background_tasks.add_task(send_password_reset_email, user["email"], raw_token)
            
            logger.info(f"🔑 Password reset initiated for {email}")

        # Always return generic success message
        return {"message": "If an account exists with this email, you will receive a password reset link shortly."}

    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        # Still return success to prevent enumeration
        return {"message": "If an account exists with this email, you will receive a password reset link shortly."}

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """
    Verify token hash and expiry, then update password.
    Invalides token after successful reset.
    """
    try:
        db = await get_db()
        token_hash = hashlib.sha256(request.token.encode()).hexdigest()
        
        # Find user with valid, non-expired token
        user = await db["users"].find_one({
            "reset_password_token": token_hash,
            "reset_password_expires": {"$gt": datetime.utcnow()}
        })
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
            
        # Hash new password
        hashed_password = get_password_hash(request.new_password)
        
        # Update password and clear reset token fields (single-use)
        await db["users"].update_one(
            {"_id": user["_id"]},
            {
                "$set": {"hashed_password": hashed_password},
                "$unset": {
                    "reset_password_token": "",
                    "reset_password_expires": ""
                }
            }
        )
        
        logger.info(f"✅ Password successfully reset for user {user['email']}")
        return {"message": "Password has been successfully updated. You can now log in."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        user = await db["users"].find_one({"email": current_user["sub"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Fallback: parse 'name' field for Google users who lack first/last
        first_name = user.get("first_name", "")
        last_name = user.get("last_name", "")
        if not first_name and not last_name and user.get("name"):
            parts = user["name"].split(" ", 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else ""
        
        username = user.get("username", "")
        if not username and user.get("email"):
            username = user["email"].split("@")[0].lower()
        
        # Generate a display-friendly short ID (e.g., E1B2C3D)
        # We use a shortened version of the Mongo ID for existing users to maintain consistency
        raw_id = str(user["_id"])
        short_id = raw_id[-7:].upper()

        return {
            "id": short_id,
            "full_id": raw_id, # Keep full ID for internal use if needed
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "email": user.get("email"),
            "role": user.get("role", "user"),
            "avatar": user.get("avatar"),
            "notifications": user.get("notifications", {"email": True, "push": False, "updates": True}),
            "email_verified": user.get("email_verified", False),
            "created_at": user.get("created_at")
        }
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/profile")
async def update_profile(data: ProfileUpdate, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        update_data = {}
        if data.firstName: update_data["first_name"] = data.firstName
        if data.lastName: update_data["last_name"] = data.lastName
        if data.username:
            # Check if username is taken
            existing = await db["users"].find_one({"username": data.username.lower(), "email": {"$ne": current_user["sub"]}})
            if existing:
                raise HTTPException(status_code=400, detail="Username already taken")
            update_data["username"] = data.username.lower()
        
        if not update_data:
            return {"message": "No changes requested"}
            
        await db["users"].update_one(
            {"email": current_user["sub"]},
            {"$set": update_data}
        )
        return {"message": "Profile updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/change-password")
async def change_password(data: PasswordChange, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        user = await db["users"].find_one({"email": current_user["sub"]})
        
        if not user or not verify_password(data.currentPassword, user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Incorrect current password")
            
        hashed_password = get_password_hash(data.newPassword)
        await db["users"].update_one(
            {"_id": user["_id"]},
            {"$set": {"hashed_password": hashed_password}}
        )
        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/notifications")
async def update_notifications(data: NotificationUpdate, current_user: dict = Depends(get_current_user)):
    try:
        db = await get_db()
        update_data = {}
        if data.email is not None: update_data["notifications.email"] = data.email
        if data.push is not None: update_data["notifications.push"] = data.push
        if data.updates is not None: update_data["notifications.updates"] = data.updates
        
        await db["users"].update_one(
            {"email": current_user["sub"]},
            {"$set": update_data}
        )
        return {"message": "Notification preferences updated"}
    except Exception as e:
        logger.error(f"Error updating notifications: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

from fastapi import File, UploadFile

@router.post("/upload-avatar")
async def upload_avatar(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Upload profile avatar to Cloudinary and update user document."""
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only JPEG, PNG, WebP and GIF images are allowed.")
        
        # Validate file size (max 2MB)
        contents = await file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size must be under 2MB.")
        
        # Upload to Cloudinary
        from services.cloudinary_service import upload_file_to_cloudinary
        result = await upload_file_to_cloudinary(contents=contents, folder="user_avatars")
        
        if not result or not result.get("url"):
            raise HTTPException(status_code=500, detail="Failed to upload image. Please try again.")
        
        avatar_url = result["url"]
        
        # Update user document
        db = await get_db()
        await db["users"].update_one(
            {"email": current_user["sub"]},
            {"$set": {"avatar": avatar_url}}
        )
        
        logger.info(f"📸 Avatar uploaded for {current_user['sub']}")
        return {"message": "Avatar uploaded successfully", "avatar_url": avatar_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Avatar upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar")
