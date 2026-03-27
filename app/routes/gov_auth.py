from fastapi import APIRouter, HTTPException, Depends, Body, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from services.mongodb_service import get_db
from core.auth import get_admin_user, require_permission
from utils.security import get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
import logging
import secrets
from bson import ObjectId

router = APIRouter(
    prefix="/gov",
    tags=["Government Portal Auth"]
)

logger = logging.getLogger(__name__)

# --- Models ---

class GovAccountCreate(BaseModel):
    name: str
    email: EmailStr
    department: str
    zip_code: str
    city: str

class GovLoginRequest(BaseModel):
    email: str
    password: str

class AccountStatusUpdate(BaseModel):
    account_id: str
    is_active: bool

# --- Endpoints ---

@router.post("/setup-account")
async def setup_gov_account(
    account: GovAccountCreate, 
    background_tasks: BackgroundTasks,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Super Admin only: Create a new government official account for the Gov Portal.
    Generates a temporary password and sends a welcome email.
    """
    db = await get_db()
    # Check if exists in government_users
    if await db["government_users"].find_one({"email": account.email.lower()}):
        raise HTTPException(status_code=400, detail="Account with this email already exists")
    
    # Generate temporary password
    temp_password = secrets.token_urlsafe(8)
    hashed_password = get_password_hash(temp_password)
    
    new_user = {
        "name": account.name,
        "email": account.email.lower(),
        "department": account.department,
        "zip_code": account.zip_code,
        "city": account.city,
        "hashed_password": hashed_password,
        # Role mapping for RBAC in Gov Portal
        "role": account.department.lower().replace(" ", "_"), 
        "created_at": datetime.utcnow(),
        "created_by": admin.get("email"),
        "is_active": True,
        "require_password_change": True
    }
    
    result = await db["government_users"].insert_one(new_user)
    
    # Send Welcome Email
    from services.email_service import send_email
    subject = f"Welcome to EAiSER Government Portal - {account.city} {account.department}"
    
    # Professional Email Template
    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; color: #333; max-width: 600px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 16px; background-color: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
        <div style="text-align: center; margin-bottom: 25px;">
            <h1 style="color: #2563eb; margin: 0; font-size: 24px;">EAiSER <span style="color: #1e293b;">Gov Portal</span></h1>
            <p style="color: #64748b; font-size: 14px; margin-top: 5px;">Official Authority Activation</p>
        </div>
        
        <div style="border-top: 2px solid #2563eb; padding-top: 20px;">
            <p>Hello <strong>{account.name}</strong>,</p>
            <p>An official administrative account has been provisioned for you to manage civic reports for:</p>
            <div style="background-color: #f8fafc; padding: 12px 20px; border-radius: 8px; font-weight: 600; color: #1e293b; display: inline-block;">
                📍 {account.city} — {account.department}
            </div>
            
            <p style="margin-top: 25px;">Your secure access credentials are detailed below:</p>
            
            <div style="background: #f3f4f6; padding: 20px; border-radius: 12px; margin: 20px 0; border: 1px solid #e2e8f0;">
                <p style="margin: 0 0 10px 0;"><strong>Portal URL:</strong> <a href="https://gov.eaiser.ai" style="color: #2563eb; text-decoration: none; font-weight: 600;">gov.eaiser.ai</a></p>
                <p style="margin: 0 0 10px 0;"><strong>Access Email:</strong> <span style="font-family: monospace;">{account.email.lower()}</span></p>
                <p style="margin: 0;"><strong>Temporary Key:</strong> <code style="background: #ffffff; padding: 4px 8px; border-radius: 6px; border: 1px dashed #cbd5e1; font-weight: 700; color: #0f172a;">{temp_password}</code></p>
            </div>
            
            <p style="font-size: 14px; color: #64748b; background-color: #fffbeb; padding: 10px; border-left: 4px solid #f59e0b; border-radius: 4px;">
                <strong>Security Protocol:</strong> You are required to change this temporary password upon your first successful authentication.
            </p>
        </div>
        
        <div style="margin-top: 35px; padding-top: 20px; border-top: 1px solid #e5e7eb; text-align: center;">
            <p style="font-size: 13px; color: #94a3b8; margin: 0;">Thank you for your service to the community.</p>
            <p style="font-size: 11px; color: #cbd5e1; margin-top: 10px;">EAiSER Administration Control Center • Internal Transmission</p>
        </div>
    </div>
    """
    
    text_content = f"""
Welcome to EAiSER Government Portal

Hello {account.name},

An official account has been created for you for {account.city} - {account.department}.

Login at: https://gov.eaiser.ai
Email: {account.email.lower()}
Temporary Password: {temp_password}

Please change your password after logging in.
"""
    
    background_tasks.add_task(send_email, account.email.lower(), subject, html_content, text_content)
    
    return {
        "success": True,
        "message": "Government account created and welcome email sent.",
        "account_id": str(result.inserted_id),
        "temp_password": temp_password # Optional: return to admin UI for immediate reference
    }

@router.get("/accounts")
async def list_gov_accounts(admin: dict = Depends(require_permission("view_authorities"))):
    """
    List all created government official accounts.
    """
    db = await get_db()
    cursor = db["government_users"].find({}, {"hashed_password": 0}).sort("created_at", -1)
    accounts = await cursor.to_list(length=100)
    for a in accounts:
        a["id"] = str(a["_id"])
        del a["_id"]
    return accounts

@router.post("/toggle-account")
async def toggle_gov_account(
    update: AccountStatusUpdate,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Activate/Deactivate a government account.
    """
    db = await get_db()
    result = await db["government_users"].update_one(
        {"_id": ObjectId(update.account_id)},
        {"$set": {"is_active": update.is_active, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Account not found")
        
    return {"success": True, "is_active": update.is_active}

@router.post("/login")
async def gov_login(creds: GovLoginRequest):
    """
    Dedicated login for the EAiSER Government Portal.
    """
    db = await get_db()
    user = await db["government_users"].find_one({"email": creds.email.lower()})
    
    if not user or not verify_password(creds.password, user["hashed_password"]):
        logger.warning(f"Failed gov login attempt for {creds.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Please contact EAiSER Admin.")
    
    # Generate token with department and zip context
    role = user.get("role", "viewer")
    token_data = {
        "sub": user["email"],
        "id": str(user["_id"]),
        "role": role,
        "dept": user.get("department"),
        "zip": user.get("zip_code"),
        "org": user.get("city"),
        "name": user.get("name"),
        "type": "gov_portal" # Identify token type
    }
    
    # Extended token life for operational use
    access_token = create_access_token(data=token_data, expires_delta=timedelta(hours=24))
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "name": user["name"],
            "email": user["email"],
            "role": role,
            "department": user.get("department"),
            "zip": user.get("zip_code"),
            "org": user.get("city"),
            "initials": "".join([n[0] for n in user["name"].split() if n]).upper()
        }
    }

@router.delete("/delete/{account_id}")
async def delete_gov_account(
    account_id: str,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Super Admin only: Delete a government official account.
    """
    db = await get_db()
    result = await db["government_users"].delete_one({"_id": ObjectId(account_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Account not found")
        
    return {"success": True, "message": "Account deleted successfully"}

@router.post("/reset-password/{account_id}")
async def reset_gov_password(
    account_id: str,
    background_tasks: BackgroundTasks,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Super Admin only: Reset a government official's password.
    Returns the new temporary password and sends a professional email notification.
    """
    db = await get_db()
    user = await db["government_users"].find_one({"_id": ObjectId(account_id)})
    
    if not user:
        raise HTTPException(status_code=404, detail="Government account not found")
        
    # Generate new secure temporary password
    temp_password = secrets.token_urlsafe(10)
    hashed_password = get_password_hash(temp_password)
    
    await db["government_users"].update_one(
        {"_id": ObjectId(account_id)},
        {"$set": {
            "hashed_password": hashed_password,
            "require_password_change": True,
            "updated_at": datetime.utcnow()
        }}
    )
    
    # --- Prepare Professional Reset Email ---
    from services.email_service import send_email
    subject = f"EAiSER Portal — Emergency Access Reset: {user.get('city')}"
    
    html_content = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px; border: 1px solid #e5e7eb; border-radius: 20px; background-color: #ffffff;">
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #2563eb; margin: 0; font-size: 24px;">EAiSER <span style="color: #0f172a;">Gov Portal</span></h1>
            <p style="color: #64748b; font-size: 13px; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px;">Security Protocol</p>
        </div>
        
        <div style="border-top: 2px solid #2563eb; padding-top: 25px;">
            <p style="font-weight: 600; font-size: 18px; color: #0f172a;">Password Reset Executed</p>
            <p style="color: #475569; line-height: 1.6;">Hello <strong>{user.get('name')}</strong>,</p>
            <p style="color: #475569; line-height: 1.6;">Per your request or an administrative action, your security credentials for the <strong>{user.get('department')}</strong> have been reset.</p>
            
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; padding: 25px; border-radius: 12px; margin: 30px 0; text-align: center;">
                <p style="margin: 0 0 10px 0; font-size: 12px; color: #64748b; text-transform: uppercase;">Your New Temporary Key</p>
                <code style="font-size: 20px; color: #1e293b; font-weight: 700; background: #ffffff; padding: 5px 15px; border-radius: 6px;">{temp_password}</code>
            </div>
            
            <p style="color: #475569; font-size: 14px;">Log in at <a href="https://gov.eaiser.ai" style="color: #2563eb; text-decoration: none; font-weight: 600;">gov.eaiser.ai</a> to finalize your access reactivation.</p>
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #f1f5f9; text-align: center;">
            <p style="font-size: 11px; color: #94a3b8; margin: 0;">EAiSER Administration Control Center • Automated Transmission</p>
        </div>
    </div>
    """
    
    text_content = f"Your EAiSER Government Portal password has been reset. New Temporary password: {temp_password}"
    
    background_tasks.add_task(send_email, user["email"], subject, html_content, text_content)
    
    return {
        "success": True, 
        "message": "Password reset successfully. Official has been notified.",
        "temp_password": temp_password
    }
