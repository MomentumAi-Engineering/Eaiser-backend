from fastapi import APIRouter, HTTPException, Depends, Body, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from services.mongodb_service import get_db
from core.auth import get_admin_user, require_permission, get_current_user
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
    role: Optional[str] = "operations" # "super_admin" or "operations"

class GovLoginRequest(BaseModel):
    email: str
    password: str
    platform: Optional[str] = None # 'app' or 'portal'

class AccountStatusUpdate(BaseModel):
    account_id: str
    is_active: bool

class ContractorCreate(BaseModel):
    company: str
    contact: str
    dept: str
    value: float
    rating: float
    status: str

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
    # 1. AUTHENTICATION & AUTHORITY CHECK
    # Only allow System Admin OR Gov Super Admin to create accounts
    is_system_admin = admin.get("type") == "admin"
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") == "super_admin"
    is_operations = admin.get("type") == "gov_portal" and admin.get("role") == "operations"

    if not is_system_admin and not is_gov_super_admin and not is_operations:
        raise HTTPException(status_code=403, detail="Unauthorized to provision accounts")

    target_role = account.role if account.role else "operations"

    # Operations Staff can ONLY create Crew Members (for their own department)
    if is_operations:
        if target_role != "crew_member":
            raise HTTPException(status_code=403, detail="Operations staff can only create Crew Members")
        if account.department != admin.get("dept"):
            raise HTTPException(status_code=403, detail=f"Can only create Crew Members for your assigned department ({admin.get('dept')})")
        account.city = admin.get("org", account.city) # Force same city

    # Gov Super Admins can only create accounts for their own City
    if is_gov_super_admin and account.city != admin.get("org"):
        raise HTTPException(status_code=403, detail="You can only provision accounts for your own city")
    
    # Gov Super Admins cannot create other Super Admins (Security safeguard)
    if is_gov_super_admin and target_role == "super_admin":
        raise HTTPException(status_code=403, detail="Only System Admins can provision new Super Admin accounts")

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
        "role": target_role, 
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
Welcome to EAiSER Platform

Hello {account.name},

An official account has been created for you ("{target_role.replace('_', ' ').title()}") for {account.city} - {account.department}.

Login via EAiSER App/Portal.
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
    List government official accounts. 
    HIERARCHICAL FILTERING:
    - System Admin sees ALL.
    - Gov Super Admin sees ONLY their city and skips other Super Admins.
    """
    db = await get_db()
    
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ["super_admin", "ops_manager"]
    is_operations = admin.get("type") == "gov_portal" and admin.get("role") == "operations"
    
    query = {}
    
    if is_gov_super_admin:
        # Gov Super Admin sees Operations Staff & Crew Members in their city
        query["city"] = admin.get("org")
        query["role"] = {"$in": ["operations", "crew_member"]}
    elif is_operations:
        # Operations Staff sees ONLY Crew Members in their specific city AND department
        query["city"] = admin.get("org")
        query["department"] = admin.get("dept")
        query["role"] = "crew_member"
    elif is_system_admin:
        # System Admin sees ALL accounts for management
        pass
    else:
        return [] # Safe fallback
    
    cursor = db["government_users"].find(query, {"hashed_password": 0}).sort("created_at", -1)
    accounts = await cursor.to_list(length=1000)
    
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
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ["super_admin", "ops_manager"]

    if not is_system_admin and not is_gov_super_admin:
        logger.warning(f"🚫 Unauthorized toggle attempt. Type: {admin.get('type')}, Role: {admin.get('role')}")
        raise HTTPException(status_code=403, detail="Unauthorized: High-level administrative clearance required.")

    db = await get_db()
    target = await db["government_users"].find_one({"_id": ObjectId(update.account_id)})
    if not target:
        raise HTTPException(status_code=404, detail="Account not found")

    # Scope check for Gov Super Admin
    if is_gov_super_admin and target.get("city") != admin.get("org"):
        raise HTTPException(status_code=403, detail="Out of jurisdiction")
    
    # Cannot toggle other super admins
    if is_gov_super_admin and target.get("role") == "super_admin" and str(target["_id"]) != admin.get("id"):
        raise HTTPException(status_code=403, detail="Cannot modify another Super Admin")

    await db["government_users"].update_one(
        {"_id": ObjectId(update.account_id)},
        {"$set": {"is_active": update.is_active, "updated_at": datetime.utcnow()}}
    )
    
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
        
    role = user.get("role", "viewer")
    req_platform = creds.platform or "unknown"
        
    # Enforce strict cross-platform boundaries with NO exceptions
    if role == "crew_member" and req_platform != "app":
        raise HTTPException(status_code=403, detail="Login Blocked: Crew members must log in via the Mobile Field App.")
    
    if role in ["operations", "super_admin", "ops_manager"] and req_platform != "portal":
        raise HTTPException(status_code=403, detail="Login Blocked: Operations/Admin staff must log in via the Web Command Center.")
        
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
            "initials": "".join([n[0] for n in user["name"].split() if n]).upper(),
            "avatar_url": user.get("avatar_url")
        }
    }

@router.delete("/delete/{account_id}")
async def delete_gov_account(
    account_id: str,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Delete a government official account.
    """
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ["super_admin", "ops_manager"]

    if not is_system_admin and not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")

    db = await get_db()
    
    # 🛡️ Safety check for ID format
    try:
        obj_id = ObjectId(account_id)
    except Exception:
         # Try finding by external 'id' field if stored (fallback)
         target = await db["government_users"].find_one({"id": account_id})
         obj_id = target["_id"] if target else None

    if not obj_id:
        raise HTTPException(status_code=400, detail="Invalid account signature")

    target = await db["government_users"].find_one({"_id": obj_id})
    if not target:
        raise HTTPException(status_code=404, detail="Identity not found in database")

    # Scope check: System Admin can do everything. Gov Super Admin only their city.
    if is_gov_super_admin:
        if target.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Target is out of regional jurisdiction")
        
        # Prevent self-deletion via this endpoint (safety)
        if target.get("email") == admin.get("sub"):
            raise HTTPException(status_code=400, detail="Self-decommission must be handled via security settings")

    await db["government_users"].delete_one({"_id": obj_id})
    return {"success": True, "message": "Identity successfully purged from command node"}

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    current_password: Optional[str] = None
    zip_code: Optional[str] = None

@router.put("/profile")
async def update_gov_profile(
    update: ProfileUpdate,
    user: dict = Depends(get_current_user)
):
    """
    Allow gov user to update their own profile and password.
    """
    db = await get_db()
    
    # If the user is trying to change password, verify the current one
    if update.password:
        if not update.current_password:
             raise HTTPException(status_code=400, detail="Current password is required to set a new password")
        
        gov_user = await db["government_users"].find_one({"email": user["email"]})
        if not verify_password(update.current_password, gov_user.get("hashed_password", "")):
             raise HTTPException(status_code=401, detail="Incorrect current password")
             
    upd = {"updated_at": datetime.utcnow()}
    if update.name: upd["name"] = update.name
    if update.zip_code: upd["zip_code"] = update.zip_code
    if update.password:
        upd["hashed_password"] = get_password_hash(update.password)
        upd["require_password_change"] = False

    await db["government_users"].update_one(
        {"email": user["email"]},
        {"$set": upd}
    )
    return {"success": True, "message": "Profile updated"}

@router.post("/profile/avatar")
async def update_gov_avatar(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    from services.cloudinary_service import upload_file_to_cloudinary
    db = await get_db()
    contents = await file.read()
    result = await upload_file_to_cloudinary(contents=contents, folder="gov_avatars")
    
    if result and result.get("url"):
        url = result.get("url")
        await db["government_users"].update_one(
            {"email": user["email"]},
            {"$set": {"avatar_url": url, "updated_at": datetime.utcnow()}}
        )
        return {"success": True, "avatar_url": url}
    raise HTTPException(status_code=500, detail="Failed to upload avatar")

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

# --- Department Management ---

class DepartmentCreate(BaseModel):
    name: str
    description: str
    icon: Optional[str] = "🏗️"

@router.post("/departments")
async def create_gov_department(
    dept: DepartmentCreate,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Create a new department in the city.
    """
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") == "super_admin"
    is_system_admin = admin.get("type") == "admin"

    if not is_system_admin and not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")

    db = await get_db()
    city = admin.get("org")
    
    # Check if exists
    if await db["gov_departments"].find_one({"city": city, "name": dept.name}):
        raise HTTPException(status_code=400, detail="Department already exists in this city")

    new_dept = {
        "name": dept.name,
        "description": dept.description,
        "icon": dept.icon,
        "city": city,
        "created_at": datetime.utcnow(),
        "created_by": admin.get("email")
    }
    
    result = await db["gov_departments"].insert_one(new_dept)
    return {"success": True, "id": str(result.inserted_id)}

@router.get("/departments")
async def list_gov_departments(admin: dict = Depends(require_permission("view_authorities"))):
    """
    List all departments for the city.
    """
    db = await get_db()
    city = admin.get("org")
    cursor = db["gov_departments"].find({"city": city}).sort("name", 1)
    depts = await cursor.to_list(length=100)
    for d in depts:
        d["id"] = str(d["_id"])
        del d["_id"]
    return depts

@router.delete("/departments/{dept_id}")
async def delete_gov_department(
    dept_id: str,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Delete a department.
    """
    db = await get_db()
    city = admin.get("org")
    result = await db["gov_departments"].delete_one({"_id": ObjectId(dept_id), "city": city})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Department not found or unauthorized")
        
    return {"success": True}

# --- Contractor Management --

@router.post("/contractors")
async def create_contractor(
    contractor: ContractorCreate,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Super Admin only: Register a new Contractor company.
    """
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ["super_admin", "ops_manager"]
    if not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Only Super Admins can onboard contractors")

    db = await get_db()
    
    new_con = contractor.dict()
    new_con["city"] = admin.get("org")
    new_con["created_at"] = datetime.utcnow()
    
    result = await db["gov_contractors"].insert_one(new_con)
    return {"id": str(result.inserted_id), "status": "Active Contractor Registered"}

@router.get("/contractors")
async def get_contractors(
    admin: dict = Depends(require_permission("view_authorities"))
):
    """
    List all contractors for the city. Can be viewed by Ops Staff and Super Admins.
    """
    db = await get_db()
    city = admin.get("org")
    
    cursor = db["gov_contractors"].find({"city": city}).sort("created_at", -1)
    contractors = await cursor.to_list(length=100)
    
    for c in contractors:
        c["id"] = str(c["_id"])
        del c["_id"]
        
    return contractors

@router.delete("/contractors/{contractor_id}")
async def delete_contractor(
    contractor_id: str,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ["super_admin", "ops_manager"]
    if not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Only Super Admins can remove contractors")

    db = await get_db()
    city = admin.get("org")
    
    result = await db["gov_contractors"].delete_one({"_id": ObjectId(contractor_id), "city": city})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Contractor not found or unauthorized")
        
    return {"success": True}
