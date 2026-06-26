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
    requested_role: Optional[str] = None  # which access door: admin / ops_manager / ops / crew


# Strict access-door → allowed account roles. A login through a given door only
# succeeds for an account whose role is in this set — so a Super Admin's
# credentials entered in the Operations Manager door are rejected outright,
# never issuing a token. Mirrors the frontend ROLE_OPTIONS.
LOGIN_DOOR_ROLES = {
    "admin": {"super_admin", "admin", "gov_admin", "access"},
    "super_admin": {"super_admin", "admin", "gov_admin", "access"},
    "ops_manager": {"ops_manager"},
    "ops": {"operations"},
    "operations": {"operations"},
    "crew": {"crew_member"},
    "crew_member": {"crew_member"},
}

DOOR_LABEL = {
    "admin": "Super Admin",
    "super_admin": "Super Admin",
    "ops_manager": "Operations Manager",
    "ops": "Operations Staff",
    "operations": "Operations Staff",
    "crew": "Field Crew",
    "crew_member": "Field Crew",
}

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
    # Uniqueness is scoped to (email + department + city): the same person may
    # belong to multiple departments, but not be added twice to the same one.
    if await db["government_users"].find_one({
        "email": account.email.lower(),
        "department": account.department,
        "city": account.city,
    }):
        raise HTTPException(status_code=400, detail="Account with this email already exists in this department")
    
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
    
    # Send the official welcome email using the SHARED branded template
    # (EAiSER logo header + MomntumAi social footer), not a one-off inline
    # template. We AWAIT it instead of firing a silent background task so the
    # response can report whether delivery actually succeeded — the silent
    # background task is exactly what hid the "email never arrived" failure.
    from services.email_service import send_gov_welcome_email
    email_sent = False
    try:
        email_sent = await send_gov_welcome_email(
            email=account.email.lower(),
            name=account.name,
            department=account.department,
            city=account.city,
            zip_code=account.zip_code,
            temporary_password=temp_password,
        )
    except Exception as e:
        logger.error(f"❌ Gov welcome email failed for {account.email.lower()}: {e}", exc_info=True)

    if not email_sent:
        logger.warning(
            f"⚠️ Account {account.email.lower()} created but welcome email was NOT delivered. "
            f"Check POSTMARK_API_TOKEN, verified sender, and that the Postmark stream is active."
        )

    return {
        "success": True,
        "message": (
            "Government account created and welcome email sent."
            if email_sent
            else "Government account created. The welcome email could not be delivered — "
                 "share the temporary password manually."
        ),
        "account_id": str(result.inserted_id),
        "email_sent": email_sent,
        "temp_password": temp_password  # returned so the admin can hand it off immediately
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
        # Gov Super Admin sees Operations Managers, Operations Staff & Crew
        # Members in their city. ops_manager MUST be included here — the
        # uniqueness check on create scopes by (email + department + city)
        # regardless of role, so leaving managers out of this list makes a
        # provisioned manager invisible while still tripping "already exists".
        query["city"] = admin.get("org")
        query["role"] = {"$in": ["ops_manager", "operations", "crew_member"]}
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
    req_platform = creds.platform or "unknown"

    # The same person can hold accounts in several departments (uniqueness is
    # scoped to email + department + city). A plain find_one({"email"}) would
    # grab an arbitrary one — usually the oldest — and reject a perfectly valid
    # password that belongs to a *different* one of those accounts. So match the
    # password against EVERY account on this email and pick the right one.
    candidates = await db["government_users"].find({"email": creds.email.lower()}).to_list(length=50)
    matched = [u for u in candidates if u.get("hashed_password") and verify_password(creds.password, u["hashed_password"])]

    if not matched:
        logger.warning(f"Failed gov login attempt for {creds.email} ({len(candidates)} account(s) on this email)")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # 🔒 STRICT ACCESS-DOOR ENFORCEMENT (backend, not bypassable).
    # If the client said which door they used (Super Admin / Ops Manager / Ops
    # Staff / Crew), only an account whose role belongs to that door may proceed.
    # This stops e.g. a Super Admin's credentials opening anything via the
    # Operations Manager door — no token is issued on a mismatch.
    if creds.requested_role:
        door = str(creds.requested_role).strip().lower()
        allowed_roles = LOGIN_DOOR_ROLES.get(door)
        if allowed_roles:
            role_matched = [u for u in matched if str(u.get("role", "")).lower() in allowed_roles]
            if not role_matched:
                label = DOOR_LABEL.get(door, "this access level")
                logger.warning(f"🚫 Door mismatch: {creds.email} tried {label} door but holds no matching role.")
                raise HTTPException(
                    status_code=403,
                    detail=f"These credentials are not authorized for {label}. Please use your correct access level.",
                )
            matched = role_matched

    # Prefer an ACTIVE account whose role fits the platform they're signing in
    # from (portal vs mobile app), so e.g. a stale crew_member duplicate never
    # blocks an Ops Manager logging into the web portal.
    def _fits_platform(u):
        r = str(u.get("role", "")).lower()
        if req_platform == "app":
            return r == "crew_member"
        return r in ["operations", "super_admin", "ops_manager", "admin"]

    active = [u for u in matched if u.get("is_active", True)]
    pool = active or matched
    user = next((u for u in pool if _fits_platform(u)), pool[0])

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Please contact EAiSER Admin.")

    role = user.get("role", "viewer")
        
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
            # `org_slug` is required by the city-side /billing page to fetch
            # subscription state. Without it the page can't identify which org
            # to query. Falls back to None for legacy users created before the
            # billing checkout flow added this field.
            "org_slug": user.get("org_slug"),
            "org_id": user.get("org_id"),
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

    # Re-send the SAME branded welcome email used at account creation (EAiSER
    # logo header + temp password card + portal CTA) so a "resend credentials"
    # action is indistinguishable from the original invite. We AWAIT it instead
    # of firing a silent background task so the response can report whether
    # delivery actually succeeded — that silent task is exactly what hid the
    # "the welcome email never arrived" failures.
    from services.email_service import send_gov_welcome_email
    email_sent = False
    try:
        email_sent = await send_gov_welcome_email(
            email=user["email"],
            name=user.get("name", "Team Member"),
            department=user.get("department", ""),
            city=user.get("city", ""),
            zip_code=user.get("zip_code", ""),
            temporary_password=temp_password,
        )
    except Exception as e:
        logger.error(f"❌ Gov credential resend failed for {user['email']}: {e}", exc_info=True)

    if not email_sent:
        logger.warning(
            f"⚠️ Password reset for {user['email']} succeeded but the email was NOT delivered. "
            f"Hand off the temporary password manually."
        )

    return {
        "success": True,
        "message": (
            "Credentials reset and welcome email re-sent."
            if email_sent
            else "Credentials reset. The welcome email could not be delivered — "
                 "share the temporary password manually."
        ),
        "email_sent": email_sent,
        "temp_password": temp_password
    }

# --- Department Management ---

class DepartmentCreate(BaseModel):
    name: str
    description: str
    icon: Optional[str] = "🏗️"
    # Canonical issue_types this department owns (drives report routing in the
    # portal). Optional so older clients keep working.
    issue_types: Optional[List[str]] = None

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
        "issue_types": dept.issue_types or [],
        "city": city,
        "created_at": datetime.utcnow(),
        "created_by": admin.get("email")
    }

    result = await db["gov_departments"].insert_one(new_dept)
    return {"success": True, "id": str(result.inserted_id)}

@router.post("/departments/seed-defaults")
async def seed_default_gov_departments(
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Idempotently create the standard set of city departments (Public Works,
    Sanitation, Water, Electrical & Traffic, Code Enforcement, Fire, Police)
    for the caller's city, each pre-mapped to the issue types it owns so
    reports route correctly. Existing departments are kept; missing ones are
    added and legacy ones get their issue_types back-filled.
    """
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") == "super_admin"
    is_system_admin = admin.get("type") in ["admin", "access"]
    if not is_system_admin and not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Only a city Super Admin can seed default departments")

    city = admin.get("org")
    if not city:
        raise HTTPException(status_code=400, detail="No city associated with this account")

    db = await get_db()
    try:
        from app.gov.default_departments import seed_default_departments
    except ImportError:
        from gov.default_departments import seed_default_departments
    touched = await seed_default_departments(db, city, admin.get("email") or "system")
    return {"success": True, "created_or_updated": touched, "count": len(touched)}

@router.get("/departments")
async def list_gov_departments(admin: dict = Depends(require_permission("view_authorities"))):
    """
    List all departments for the city.

    For a city Super Admin we also lazily ensure the standard default
    departments exist (idempotent) — this back-fills every already-existing city
    on its next portal load, so admins don't have to seed manually. Ops/crew
    reads never mutate.
    """
    db = await get_db()
    city = admin.get("org")

    is_super = admin.get("type") == "gov_portal" and admin.get("role") == "super_admin"
    is_system_admin = admin.get("type") in ["admin", "access"]
    if city and (is_super or is_system_admin):
        try:
            try:
                from app.gov.default_departments import seed_default_departments
            except ImportError:
                from gov.default_departments import seed_default_departments
            await seed_default_departments(db, city, admin.get("email") or "system")
        except Exception as e:
            logger.warning(f"⚠️ Default-department auto-seed skipped for '{city}': {e}")

    cursor = db["gov_departments"].find({"city": city}).sort("name", 1)
    depts = await cursor.to_list(length=100)
    for d in depts:
        d["id"] = str(d["_id"])
        del d["_id"]
    return depts

class DepartmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    issue_types: Optional[List[str]] = None

@router.put("/departments/{dept_id}")
async def update_gov_department(
    dept_id: str,
    update: DepartmentUpdate,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Edit a department (name / description / color / issue_types). Scoped to the
    caller's city. Only provided fields are changed.
    """
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") == "super_admin"
    is_system_admin = admin.get("type") in ["admin", "access"]
    if not is_system_admin and not is_gov_super_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")

    db = await get_db()
    city = admin.get("org")

    target = await db["gov_departments"].find_one({"_id": ObjectId(dept_id), "city": city})
    if not target:
        raise HTTPException(status_code=404, detail="Department not found or unauthorized")

    changes = {}
    if update.name is not None and update.name.strip():
        new_name = update.name.strip()
        # Block renaming onto another existing department in the same city.
        clash = await db["gov_departments"].find_one({
            "city": city,
            "name": new_name,
            "_id": {"$ne": ObjectId(dept_id)},
        })
        if clash:
            raise HTTPException(status_code=400, detail="Another department already uses that name")
        changes["name"] = new_name
    if update.description is not None:
        changes["description"] = update.description.strip()
    if update.icon is not None:
        changes["icon"] = update.icon
    if update.issue_types is not None:
        changes["issue_types"] = update.issue_types

    if not changes:
        return {"success": True, "updated": 0}

    changes["updated_at"] = datetime.utcnow()
    await db["gov_departments"].update_one({"_id": ObjectId(dept_id)}, {"$set": changes})
    return {"success": True, "updated": 1}

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
