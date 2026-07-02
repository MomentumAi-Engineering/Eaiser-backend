from fastapi import APIRouter, HTTPException, Depends, Body, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from services.mongodb_service import get_db
from core.auth import get_admin_user, require_permission, get_current_user
from core.audit import record_audit
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
# credentials entered in the Staff door are rejected outright, never issuing a
# token. Mirrors the frontend ROLE_OPTIONS.
#
# Gov role hierarchy (top → bottom):
#   • Super Admin       (super_admin)      — City head (Mayor / City Manager), city-wide
#   • Operation Manager (ops_manager)      — head over ALL departments, city-wide ops
#   • Department Admin  (department_admin) — head of ONE department
#   • Staff             (operations)       — works inside a department
#   • Field Crew        (crew_member)      — mobile field app only
# Legacy door ids (`ops` → Staff) are kept as aliases so older clients keep working.
LOGIN_DOOR_ROLES = {
    "admin": {"super_admin", "admin", "gov_admin", "access"},
    "super_admin": {"super_admin", "admin", "gov_admin", "access"},
    # IT Super Admin — SYSTEM door (create cities, billing, cross-city audit)
    "it_super_admin": {"it_super_admin"},
    "it_admin": {"it_super_admin"},
    "system_admin": {"it_super_admin"},
    # Operation Manager — head over all departments
    "ops_manager": {"ops_manager"},
    "operations_manager": {"ops_manager"},
    # Department Admin — head of one department
    "department_admin": {"department_admin"},
    "dept_admin": {"department_admin"},
    "dept_head": {"department_admin"},
    # Staff — works inside a department
    "staff": {"operations"},
    "dept_staff": {"operations"},
    "ops": {"operations"},           # legacy door id
    "operations": {"operations"},
    # Mayor / Council — read-only executive view
    "mayor": {"mayor"},
    "council": {"mayor"},
    # Onboarding Specialist — read-only setup reviewer (guides City Manager setup)
    "onboarding_specialist": {"onboarding_specialist"},
    "specialist": {"onboarding_specialist"},
    "onboarding": {"onboarding_specialist"},
    # Field crew
    "crew": {"crew_member"},
    "crew_member": {"crew_member"},
}

DOOR_LABEL = {
    "admin": "Super Admin",
    "super_admin": "Super Admin",
    "it_super_admin": "IT Super Admin",
    "it_admin": "IT Super Admin",
    "system_admin": "IT Super Admin",
    "ops_manager": "Operation Manager",
    "operations_manager": "Operation Manager",
    "department_admin": "Department Admin",
    "dept_admin": "Department Admin",
    "dept_head": "Department Admin",
    "staff": "Staff",
    "dept_staff": "Staff",
    "ops": "Staff",
    "operations": "Staff",
    "mayor": "Mayor / Council",
    "council": "Mayor / Council",
    "onboarding_specialist": "Onboarding Specialist",
    "specialist": "Onboarding Specialist",
    "onboarding": "Onboarding Specialist",
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

# Self-service contractor onboarding for the Field App ("apply to work").
class ContractorRegister(BaseModel):
    company: str
    contact: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    dept: Optional[str] = None      # department they're applying to work for
    city: Optional[str] = None      # city/jurisdiction (resolved from the ZIP)
    zip_code: Optional[str] = None  # the area ZIP they're applying for

class ContractorLogin(BaseModel):
    email: str
    password: str

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
    role = str(admin.get("role", "")).lower()
    is_system_admin = admin.get("type") == "admin"
    is_gov_super_admin = admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin")
    # Operation Manager — head over ALL departments (city-wide ops). Provisions
    # Department Admins, Staff and Crew anywhere in their city.
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    # Department Admin — head of ONE department. Provisions Staff + Crew for that
    # department only.
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    is_operations = admin.get("type") == "gov_portal" and role == "operations"

    if not (is_system_admin or is_gov_super_admin or is_ops_manager or is_dept_admin or is_operations):
        raise HTTPException(status_code=403, detail="Unauthorized to provision accounts")

    target_role = account.role if account.role else "operations"

    # Operation Manager can create Department Admins / Staff / Crew across their city.
    if is_ops_manager:
        if target_role not in ("department_admin", "operations", "crew_member"):
            raise HTTPException(status_code=403, detail="Operation Manager can only create Department Admins, Staff or Crew")
        account.city = admin.get("org", account.city)  # Force same city

    # Department Admin can create Staff + Crew, but only inside their own department.
    if is_dept_admin:
        if target_role not in ("operations", "crew_member"):
            raise HTTPException(status_code=403, detail="Department Admin can only create Staff or Crew members")
        if account.department != admin.get("dept"):
            raise HTTPException(status_code=403, detail=f"You can only add people to your own department ({admin.get('dept')})")
        account.city = admin.get("org", account.city)  # Force same city

    # Staff can ONLY create Crew Members (for their own department)
    if is_operations:
        if target_role != "crew_member":
            raise HTTPException(status_code=403, detail="Staff can only create Crew Members")
        if account.department != admin.get("dept"):
            raise HTTPException(status_code=403, detail=f"Can only create Crew Members for your assigned department ({admin.get('dept')})")
        account.city = admin.get("org", account.city) # Force same city

    # A CITY Super Admin (gov-portal) can only create accounts for their OWN city.
    # System admins (type "admin") and IT Super Admins are platform-level and may
    # provision into any city (incl. the SYSTEM scope) — so they're exempt here.
    is_city_super_admin = admin.get("type") == "gov_portal" and role == "super_admin"
    if is_city_super_admin and account.city != admin.get("org"):
        raise HTTPException(status_code=403, detail="You can only provision accounts for your own city")

    # A CITY Super Admin cannot mint another (City) Super Admin — only an IT Super
    # Admin / System can. (The IT Super Admin sits above the city Super Admin.)
    if is_city_super_admin and target_role == "super_admin":
        raise HTTPException(status_code=403, detail="Only an IT Super Admin can provision new City Super Admin accounts")

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

    await record_audit(
        admin, "account_created", target=account.email.lower(),
        detail=f"{target_role} · {account.department} · {account.city}",
        city=account.city, meta={"role": target_role, "department": account.department},
    )

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

    role = str(admin.get("role", "")).lower()
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin")
    # Operation Manager oversees ALL departments; Department Admin only its own.
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    is_operations = admin.get("type") == "gov_portal" and role == "operations"

    query = {}

    if is_gov_super_admin:
        # City head sees every Operation Manager, Department Admin, Staff & Crew
        # in their city. (Uniqueness on create scopes by email+department+city
        # regardless of role, so all roles must appear here.)
        query["city"] = admin.get("org")
        query["role"] = {"$in": ["ops_manager", "department_admin", "operations", "crew_member", "mayor"]}
    elif is_ops_manager:
        # Operation Manager sees every Department Admin, Staff & Crew in the city.
        query["city"] = admin.get("org")
        query["role"] = {"$in": ["department_admin", "operations", "crew_member"]}
    elif is_dept_admin:
        # Department Admin sees the Staff & Crew of their OWN department only.
        query["city"] = admin.get("org")
        query["department"] = admin.get("dept")
        query["role"] = {"$in": ["operations", "crew_member"]}
    elif is_operations:
        # Staff sees ONLY Crew Members in their specific city AND department
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
    role = str(admin.get("role", "")).lower()
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin")
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"

    if not (is_system_admin or is_gov_super_admin or is_ops_manager or is_dept_admin):
        logger.warning(f"🚫 Unauthorized toggle attempt. Type: {admin.get('type')}, Role: {admin.get('role')}")
        raise HTTPException(status_code=403, detail="Unauthorized: High-level administrative clearance required.")

    db = await get_db()
    target = await db["government_users"].find_one({"_id": ObjectId(update.account_id)})
    if not target:
        raise HTTPException(status_code=404, detail="Account not found")

    target_role = str(target.get("role", "")).lower()

    # City Super Admin: same-city only, and cannot touch another Super Admin.
    if is_gov_super_admin:
        if target.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Out of jurisdiction")
        if target_role == "super_admin" and str(target["_id"]) != admin.get("id"):
            raise HTTPException(status_code=403, detail="Cannot modify another Super Admin")

    # Operation Manager: city-wide, but cannot touch a Super Admin or another Operation Manager.
    if is_ops_manager:
        if target.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Out of jurisdiction")
        if target_role in ("super_admin", "ops_manager") and str(target["_id"]) != admin.get("id"):
            raise HTTPException(status_code=403, detail="Operation Manager cannot modify a Super Admin or another Operation Manager")

    # Department Admin: only Staff/Crew, and only within their own city + department.
    if is_dept_admin:
        if target.get("city") != admin.get("org") or target.get("department") != admin.get("dept"):
            raise HTTPException(status_code=403, detail="Out of jurisdiction: you can only manage your own department")
        if target_role not in ("operations", "crew_member"):
            raise HTTPException(status_code=403, detail="Department Admin can only manage Staff and Crew")

    await db["government_users"].update_one(
        {"_id": ObjectId(update.account_id)},
        {"$set": {"is_active": update.is_active, "updated_at": datetime.utcnow()}}
    )

    await record_audit(
        admin, "account_activated" if update.is_active else "account_deactivated",
        target=target.get("email"), detail=f"{target.get('role')} · {target.get('department')}",
        city=target.get("city"),
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
        return r in ["operations", "super_admin", "it_super_admin", "ops_manager", "admin", "department_admin", "mayor", "onboarding_specialist"]

    active = [u for u in matched if u.get("is_active", True)]
    pool = active or matched
    user = next((u for u in pool if _fits_platform(u)), pool[0])

    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Please contact EAiSER Admin.")

    role = user.get("role", "viewer")
        
    # Enforce strict cross-platform boundaries with NO exceptions
    if role == "crew_member" and req_platform != "app":
        raise HTTPException(status_code=403, detail="Login Blocked: Crew members must log in via the Mobile Field App.")
    
    if role in ["operations", "super_admin", "it_super_admin", "ops_manager", "department_admin", "mayor", "onboarding_specialist"] and req_platform != "portal":
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
    
    # Some legacy accounts have no `name` field — fall back to the email prefix so
    # login never 500s on a missing key.
    display_name = user.get("name") or (user.get("email", "").split("@")[0]) or "User"
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "name": display_name,
            "email": user.get("email", ""),
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
            "initials": ("".join([n[0] for n in display_name.split() if n]).upper()[:2]) or "U",
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
    role = str(admin.get("role", "")).lower()
    is_system_admin = admin.get("type") in ["admin", "access"]
    is_gov_super_admin = admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin")
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"

    if not (is_system_admin or is_gov_super_admin or is_ops_manager or is_dept_admin):
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

    target_role = str(target.get("role", "")).lower()

    # Scope check: System Admin can do everything. Gov Super Admin only their city.
    if is_gov_super_admin:
        if target.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Target is out of regional jurisdiction")

        # Prevent self-deletion via this endpoint (safety)
        if target.get("email") == admin.get("sub"):
            raise HTTPException(status_code=400, detail="Self-decommission must be handled via security settings")

    # Operation Manager: city-wide, but cannot remove a Super Admin or another Operation Manager.
    if is_ops_manager:
        if target.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Target is out of regional jurisdiction")
        if target_role in ("super_admin", "ops_manager") and target.get("email") != admin.get("sub"):
            raise HTTPException(status_code=403, detail="Operation Manager cannot remove a Super Admin or another Operation Manager")
        if target.get("email") == admin.get("sub"):
            raise HTTPException(status_code=400, detail="Self-decommission must be handled via security settings")

    # Department Admin: only Staff/Crew, only within their own city + department.
    if is_dept_admin:
        if target.get("city") != admin.get("org") or target.get("department") != admin.get("dept"):
            raise HTTPException(status_code=403, detail="Target is outside your department")
        if target_role not in ("operations", "crew_member"):
            raise HTTPException(status_code=403, detail="Department Admin can only remove Staff and Crew")
        if target.get("email") == admin.get("sub"):
            raise HTTPException(status_code=400, detail="Self-decommission must be handled via security settings")

    await db["government_users"].delete_one({"_id": obj_id})
    await record_audit(
        admin, "account_deleted", target=target.get("email"),
        detail=f"{target_role} · {target.get('department')} · {target.get('city')}",
        city=target.get("city"),
    )
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

    # The same email can hold several accounts (uniqueness is email+dept+city).
    # Matching by email alone grabs an arbitrary account, so the current-password
    # check could verify against the WRONG account and 401 a valid password.
    # Always target THIS account via the token's account id.
    acct_filter = {"_id": ObjectId(user["id"])} if user.get("id") else {"email": user["email"]}

    # If the user is trying to change password, verify the current one
    if update.password:
        if not update.current_password:
             raise HTTPException(status_code=400, detail="Current password is required to set a new password")

        gov_user = await db["government_users"].find_one(acct_filter)
        if not gov_user or not verify_password(update.current_password, gov_user.get("hashed_password", "")):
             raise HTTPException(status_code=401, detail="Incorrect current password")

    upd = {"updated_at": datetime.utcnow()}
    if update.name: upd["name"] = update.name
    if update.zip_code: upd["zip_code"] = update.zip_code
    if update.password:
        upd["hashed_password"] = get_password_hash(update.password)
        upd["require_password_change"] = False

    await db["government_users"].update_one(acct_filter, {"$set": upd})
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
        acct_filter = {"_id": ObjectId(user["id"])} if user.get("id") else {"email": user["email"]}
        await db["government_users"].update_one(
            acct_filter,
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

    # Scope reset rights below Super Admin / system admin (which are unrestricted):
    #   • Operation Manager — any Staff/Crew/Department Admin in their city
    #   • Department Admin   — only Staff/Crew inside their own department
    role = str(admin.get("role", "")).lower()
    target_role = str(user.get("role", "")).lower()
    if admin.get("type") == "gov_portal" and role == "ops_manager":
        if user.get("city") != admin.get("org"):
            raise HTTPException(status_code=403, detail="Target is outside your city")
        if target_role in ("super_admin", "ops_manager"):
            raise HTTPException(status_code=403, detail="Operation Manager cannot reset a Super Admin or another Operation Manager")
    elif admin.get("type") == "gov_portal" and role == "department_admin":
        if user.get("city") != admin.get("org") or user.get("department") != admin.get("dept"):
            raise HTTPException(status_code=403, detail="Target is outside your department")
        if target_role not in ("operations", "crew_member"):
            raise HTTPException(status_code=403, detail="Department Admin can only reset Staff and Crew credentials")

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

    await record_audit(
        admin, "password_reset", target=user.get("email"),
        detail=f"Credentials reset for {user.get('role')}", city=user.get("city"),
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
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ("super_admin", "it_super_admin")
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
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ("super_admin", "it_super_admin")
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

    is_super = admin.get("type") == "gov_portal" and admin.get("role") in ("super_admin", "it_super_admin")
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
    is_gov_super_admin = admin.get("type") == "gov_portal" and admin.get("role") in ("super_admin", "it_super_admin")
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

# Budget ceiling (USD) a Department Admin can approve a contractor up to. Above
# this, approval auto-escalates to the City Manager (Super Admin). Mirrors the
# architecture diagram's "Dept Head approves within budget; above → escalate".
DEPT_HEAD_APPROVAL_LIMIT = 25000

class ContractorDecision(BaseModel):
    note: Optional[str] = None

@router.post("/contractors")
async def create_contractor(
    contractor: ContractorCreate,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Onboard / invite a contractor. The approval lifecycle depends on who creates it:
      • Super Admin / Operation Manager  → approved immediately.
      • Department Admin                 → approved if within their budget limit,
                                           else auto-escalated to the City Manager.
      • Staff                            → pending (awaits Department Admin approval).
    Department-scoped creators (Dept Admin / Staff) pin the contractor to their dept.
    """
    role = str(admin.get("role", "")).lower()
    is_super = admin.get("type") in ["admin", "access"] or (admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin"))
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    is_staff = admin.get("type") == "gov_portal" and role == "operations"

    if not (is_super or is_ops_manager or is_dept_admin or is_staff):
        raise HTTPException(status_code=403, detail="Not authorized to onboard contractors")

    db = await get_db()

    new_con = contractor.dict()
    new_con["city"] = admin.get("org")
    # Dept-scoped creators can only register a contractor for their own department.
    if is_dept_admin or is_staff:
        new_con["dept"] = admin.get("dept") or new_con.get("dept")
    new_con["created_at"] = datetime.utcnow()
    new_con["created_by"] = admin.get("email")
    new_con["created_by_role"] = role or admin.get("type")

    value = float(new_con.get("value") or 0)
    now = datetime.utcnow()
    if is_super or is_ops_manager:
        new_con["approval_status"] = "approved"
        new_con["approved_by"] = admin.get("email")
        new_con["approved_at"] = now
    elif is_dept_admin:
        if value > DEPT_HEAD_APPROVAL_LIMIT:
            new_con["approval_status"] = "escalated"
            new_con["escalated_by"] = admin.get("email")
            new_con["escalated_at"] = now
        else:
            new_con["approval_status"] = "approved"
            new_con["approved_by"] = admin.get("email")
            new_con["approved_at"] = now
    else:  # staff invites
        new_con["approval_status"] = "pending"

    result = await db["gov_contractors"].insert_one(new_con)
    return {
        "id": str(result.inserted_id),
        "approval_status": new_con["approval_status"],
        "escalated": new_con["approval_status"] == "escalated",
    }

@router.get("/contractors")
async def get_contractors(
    admin: dict = Depends(require_permission("view_authorities"))
):
    """
    List contractors for the caller's city. Super Admin / Operation Manager see
    every department's contractors; a Department Admin or Staff member only sees
    their own department's.
    """
    db = await get_db()
    city = admin.get("org")
    role = str(admin.get("role", "")).lower()

    query = {"city": city}
    if admin.get("type") == "gov_portal" and role in ("department_admin", "operations"):
        query["dept"] = admin.get("dept")

    cursor = db["gov_contractors"].find(query).sort("created_at", -1)
    contractors = await cursor.to_list(length=100)

    for c in contractors:
        c["id"] = str(c["_id"])
        del c["_id"]
        # Legacy rows created before the approval workflow have no status — treat
        # them as already approved so they keep appearing as assignable.
        c.setdefault("approval_status", "approved")

    return contractors

async def _load_contractor_for_decision(db, contractor_id, admin, is_dept_admin):
    """Fetch a contractor in the caller's city and enforce dept scope for a Dept Admin."""
    con = await db["gov_contractors"].find_one({"_id": ObjectId(contractor_id), "city": admin.get("org")})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")
    if is_dept_admin and con.get("dept") != admin.get("dept"):
        raise HTTPException(status_code=403, detail="Contractor is outside your department")
    return con

@router.post("/contractors/{contractor_id}/approve")
async def approve_contractor(
    contractor_id: str,
    decision: ContractorDecision = Body(default=ContractorDecision()),
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """
    Approve a contractor. A Department Admin can approve only within their budget
    limit — above it, the request auto-escalates to the City Manager instead.
    An Operation Manager / Super Admin (City Manager) can approve any amount,
    which is how escalated contractors get cleared.
    """
    role = str(admin.get("role", "")).lower()
    is_super = admin.get("type") in ["admin", "access"] or (admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin"))
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    if not (is_super or is_ops_manager or is_dept_admin):
        raise HTTPException(status_code=403, detail="Not authorized to approve contractors")

    db = await get_db()
    con = await _load_contractor_for_decision(db, contractor_id, admin, is_dept_admin)
    value = float(con.get("value") or 0)
    now = datetime.utcnow()

    if is_dept_admin and value > DEPT_HEAD_APPROVAL_LIMIT:
        await db["gov_contractors"].update_one(
            {"_id": con["_id"]},
            {"$set": {"approval_status": "escalated", "escalated_by": admin.get("email"), "escalated_at": now}},
        )
        return {
            "success": True,
            "approval_status": "escalated",
            "message": f"${value:,.0f} exceeds your ${DEPT_HEAD_APPROVAL_LIMIT:,} approval limit — escalated to the City Manager.",
        }

    await db["gov_contractors"].update_one(
        {"_id": con["_id"]},
        {"$set": {"approval_status": "approved", "approved_by": admin.get("email"), "approved_at": now}},
    )
    await record_audit(
        admin, "contractor_approved", target=con.get("company"),
        detail=f"{con.get('company')} · ${value:,.0f} · {con.get('dept')}", city=con.get("city"),
    )
    return {"success": True, "approval_status": "approved"}

@router.post("/contractors/{contractor_id}/escalate")
async def escalate_contractor(
    contractor_id: str,
    decision: ContractorDecision = Body(default=ContractorDecision()),
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """Escalate a contractor to the City Manager for approval (Dept Admin / Operation Manager)."""
    role = str(admin.get("role", "")).lower()
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    if not (is_dept_admin or is_ops_manager):
        raise HTTPException(status_code=403, detail="Only a Department Admin or Operation Manager can escalate")

    db = await get_db()
    con = await _load_contractor_for_decision(db, contractor_id, admin, is_dept_admin)
    await db["gov_contractors"].update_one(
        {"_id": con["_id"]},
        {"$set": {"approval_status": "escalated", "escalated_by": admin.get("email"), "escalated_at": datetime.utcnow(), "escalation_note": decision.note}},
    )
    await record_audit(
        admin, "contractor_escalated", target=con.get("company"),
        detail=f"{con.get('company')} escalated to City Manager", city=con.get("city"),
    )
    return {"success": True, "approval_status": "escalated"}

@router.post("/contractors/{contractor_id}/reject")
async def reject_contractor(
    contractor_id: str,
    decision: ContractorDecision = Body(default=ContractorDecision()),
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """Reject a contractor (Dept Admin within their dept, or Operation Manager / Super Admin)."""
    role = str(admin.get("role", "")).lower()
    is_super = admin.get("type") in ["admin", "access"] or (admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin"))
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    if not (is_super or is_ops_manager or is_dept_admin):
        raise HTTPException(status_code=403, detail="Not authorized to reject contractors")

    db = await get_db()
    con = await _load_contractor_for_decision(db, contractor_id, admin, is_dept_admin)
    await db["gov_contractors"].update_one(
        {"_id": con["_id"]},
        {"$set": {"approval_status": "rejected", "rejected_by": admin.get("email"), "rejected_at": datetime.utcnow(), "rejection_note": decision.note}},
    )
    await record_audit(
        admin, "contractor_rejected", target=con.get("company"),
        detail=(decision.note or f"{con.get('company')} rejected"), city=con.get("city"),
    )
    return {"success": True, "approval_status": "rejected"}

class ContractorInvite(BaseModel):
    email: EmailStr
    company: Optional[str] = None
    dept: Optional[str] = None
    zip_code: Optional[str] = None
    note: Optional[str] = None

@router.post("/contractors/invite")
async def invite_contractor(payload: ContractorInvite, admin: dict = Depends(require_permission("manage_authorities"))):
    """Staff / Dept Head / City Manager invites a contractor to APPLY to work.
    Creates an 'invited' contractor stub (so it shows in the portal) and emails
    the contractor a link to register + submit compliance in the Field App."""
    db = await get_db()
    city = admin.get("org")
    dept = payload.dept or admin.get("dept")
    email = payload.email.lower()

    existing = await db["gov_contractors"].find_one({"email": email, "city": city})
    if existing:
        # Never downgrade an already-active contractor back to "invited".
        if str(existing.get("approval_status", "")).lower() in ("approved", "pending", "escalated"):
            raise HTTPException(status_code=400, detail="This contractor already exists for your city.")
        await db["gov_contractors"].update_one(
            {"_id": existing["_id"]},
            {"$set": {
                "approval_status": "invited", "invited_by": admin.get("email"),
                "invited_at": datetime.utcnow(), "dept": dept, "zip_code": payload.zip_code,
            }},
        )
        con_id = str(existing["_id"])
    else:
        res = await db["gov_contractors"].insert_one({
            "company": payload.company or email.split("@")[0],
            "contact": payload.company or "",
            "email": email,
            "dept": dept,
            "city": city,
            "zip_code": payload.zip_code,
            "value": 0,
            "rating": 0,
            "approval_status": "invited",
            "invited_by": admin.get("email"),
            "invited_at": datetime.utcnow(),
            "created_at": datetime.utcnow(),
        })
        con_id = str(res.inserted_id)

    email_sent = False
    try:
        from services.email_service import send_contractor_invite_email
        email_sent = await send_contractor_invite_email(
            contractor_email=email,
            company=payload.company or "",
            city=city or "your city",
            department=dept or "",
            zip_code=payload.zip_code or "",
            invited_by=admin.get("name") or admin.get("email") or "",
        )
    except Exception as e:
        logger.error(f"contractor invite email failed for {email}: {e}", exc_info=True)

    await record_audit(
        admin, "contractor_invited", target=email,
        detail=f"Invited {payload.company or email} to apply ({dept or 'any dept'})", city=city,
    )
    return {
        "success": True,
        "contractor_id": con_id,
        "email_sent": email_sent,
        "message": "Invitation sent." if email_sent else
                   "Contractor marked as invited, but the email could not be delivered.",
    }

@router.delete("/contractors/{contractor_id}")
async def delete_contractor(
    contractor_id: str,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    role = str(admin.get("role", "")).lower()
    is_super = admin.get("type") in ["admin", "access"] or (admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin"))
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    if not (is_super or is_ops_manager or is_dept_admin):
        raise HTTPException(status_code=403, detail="Not authorized to remove contractors")

    db = await get_db()
    city = admin.get("org")

    # A Department Admin can only remove a contractor from their own department.
    query = {"_id": ObjectId(contractor_id), "city": city}
    if is_dept_admin:
        query["dept"] = admin.get("dept")

    result = await db["gov_contractors"].delete_one(query)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Contractor not found or unauthorized")

    return {"success": True}

# --- Contractor Field App: self-service onboarding + work orders ---

def _contractor_public(con):
    return {
        "id": str(con["_id"]),
        "company": con.get("company"),
        "contact": con.get("contact"),
        "email": con.get("email"),
        "phone": con.get("phone"),
        "dept": con.get("dept"),
        "city": con.get("city"),
        "zip_code": con.get("zip_code"),
        "approval_status": con.get("approval_status", "pending"),
        "rating": con.get("rating", 0),
    }

@router.get("/contractor/areas")
async def contractor_areas_by_zip(zip: str = ""):
    """
    Public lookup for the Field App "apply to work" flow. Given a ZIP, find the
    city that services it and the departments active there, so a contractor PICKS
    from a real list instead of typing free text — guaranteeing their city/dept
    match the gov org so the right Department Admin sees the application.

    Also returns a `directory` of every city that has departments, so if the ZIP
    can't be resolved the contractor can still pick their city manually.
    """
    db = await get_db()
    z = str(zip or "").strip()

    def _fmt(rows):
        return [{"id": str(d["_id"]), "name": d.get("name"), "icon": d.get("icon")} for d in rows]

    # 1) Resolve the servicing city for this ZIP from whichever source has it.
    city = None
    if z:
        org = await db["organizations"].find_one({"$or": [{"zip_codes": z}, {"primary_zip": z}]})
        if org:
            city = org.get("legal_name")
        if not city:
            acct = await db["government_users"].find_one({"zip_code": z})
            if acct:
                city = acct.get("city")

    departments = []
    if city:
        rows = await db["gov_departments"].find({"city": city}).sort("name", 1).to_list(length=100)
        departments = _fmt(rows)

    # 2) Directory of every city that has departments (manual-pick fallback).
    directory = {}
    all_rows = await db["gov_departments"].find({}).sort("name", 1).to_list(length=500)
    for d in all_rows:
        c = d.get("city")
        if not c:
            continue
        directory.setdefault(c, []).append({"id": str(d["_id"]), "name": d.get("name"), "icon": d.get("icon")})
    directory_list = [{"city": c, "departments": deps} for c, deps in sorted(directory.items())]

    return {
        "zip": z,
        "city": city,
        "departments": departments,
        "count": len(departments),
        "directory": directory_list,
    }

@router.post("/contractor/register")
async def contractor_register(reg: ContractorRegister):
    """
    Self-service contractor signup from the Field App ("apply to work"). Creates a
    PENDING contractor that a Department Admin / City Manager must approve before
    it can receive work orders. Reuses the existing contractor approval workflow.
    """
    db = await get_db()
    email = reg.email.lower()
    if await db["gov_contractors"].find_one({"email": email}):
        raise HTTPException(status_code=400, detail="A contractor account with this email already exists")

    doc = {
        "company": reg.company,
        "contact": reg.contact,
        "email": email,
        "hashed_password": get_password_hash(reg.password),
        "phone": reg.phone,
        "dept": reg.dept or "",
        "city": reg.city or "",
        "zip_code": reg.zip_code or "",
        "value": 0,
        "rating": 0,
        "status": "Applied",
        "approval_status": "pending",   # awaits Dept Head review
        "created_by": email,
        "created_by_role": "self",
        "is_active": True,
        "require_password_change": False,
        "created_at": datetime.utcnow(),
    }
    result = await db["gov_contractors"].insert_one(doc)
    return {"success": True, "id": str(result.inserted_id), "approval_status": "pending"}

@router.post("/contractor/login")
async def contractor_login(creds: ContractorLogin):
    """Field App login for a self-registered contractor (separate from crew/gov login)."""
    db = await get_db()
    con = await db["gov_contractors"].find_one({"email": creds.email.lower()})
    if not con or not con.get("hashed_password") or not verify_password(creds.password, con["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not con.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Please contact the city.")

    token_data = {
        "sub": con["email"],
        "id": str(con["_id"]),
        "role": "contractor",
        "dept": con.get("dept"),
        "org": con.get("city"),
        "name": con.get("company"),
        "type": "contractor",
    }
    access_token = create_access_token(data=token_data, expires_delta=timedelta(hours=24))
    return {"access_token": access_token, "token_type": "bearer", "contractor": _contractor_public(con)}

@router.get("/contractor/me")
async def contractor_me(user: dict = Depends(get_current_user)):
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con = await db["gov_contractors"].find_one({"_id": ObjectId(user.get("id"))})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")
    return _contractor_public(con)

def _work_order_view(r):
    created = r.get("created_at")
    # Description can live at the top level or nested under the AI report.
    desc = r.get("description") or ((r.get("report") or {}).get("issue_overview") or {}).get("summary_explanation") or ""
    # Severity may be a plain string or a {label,...} dict.
    sev = r.get("severity") or r.get("risk")
    if isinstance(sev, dict):
        sev = sev.get("label")
    # The citizen's report photo IS the "before" — first cloudinary image.
    citizen_img = None
    iu = r.get("image_urls")
    if isinstance(iu, list) and iu:
        citizen_img = iu[0]
    citizen_img = citizen_img or r.get("image_url") or r.get("media_url") or r.get("image")
    return {
        "id": str(r.get("_id")),
        "report_id": r.get("report_id") or r.get("issue_id") or str(r.get("_id")),
        "type": r.get("issue_type") or r.get("category") or "Issue",
        "location": r.get("address") or r.get("location") or "",
        "zip": r.get("zip_code"),
        "status": r.get("status"),
        "severity": str(sev).title() if sev else None,
        "priority": r.get("priority"),
        # NOTE: citizen/reporter identity is intentionally NOT exposed to the
        # contractor — only the location, severity and photos they need to work.
        # `image` = citizen report photo (the "before"). `after_photo_url` = the
        # contractor's completion photo, sent for staff verification.
        "image": citizen_img,
        "after_photo_url": r.get("resolution_media_url"),
        "coordinates": [r.get("latitude") or 0, r.get("longitude") or 0],
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "assigned_at": (r.get("assigned_at").isoformat() if hasattr(r.get("assigned_at"), "isoformat") else r.get("assigned_at")),
        "description": desc,
        "accepted": bool(r.get("contractor_accepted")),
        "quote": r.get("contractor_quote"),
    }

@router.get("/contractor/work-orders")
async def contractor_work_orders(user: dict = Depends(get_current_user)):
    """Reports assigned to this contractor — their work orders. Only approved contractors get any."""
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con = await db["gov_contractors"].find_one({"_id": ObjectId(user.get("id"))})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")
    if str(con.get("approval_status", "")).lower() != "approved":
        return {"approval_status": con.get("approval_status", "pending"), "work_orders": []}

    # Match by id OR email so re-created contractor accounts still see their work.
    match = {"$or": [{"contractor_ref": str(con["_id"])}]}
    if con.get("email"):
        match["$or"].append({"contractor_email": con.get("email")})
    cursor = db["issues"].find(match).sort("assigned_at", -1)
    rows = await cursor.to_list(length=200)
    return {"approval_status": "approved", "work_orders": [_work_order_view(r) for r in rows]}

async def _contractor_owns_report(db, user, report_id):
    con = await db["gov_contractors"].find_one({"_id": ObjectId(user.get("id"))})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")
    # Reports may carry a custom string _id (e.g. "1NANH1Q") OR an ObjectId —
    # try the raw value first, then fall back to ObjectId.
    rep = await db["issues"].find_one({"_id": report_id})
    if not rep:
        try:
            rep = await db["issues"].find_one({"_id": ObjectId(report_id)})
        except Exception:
            rep = None
    if not rep:
        raise HTTPException(status_code=404, detail="Work order not found")
    owns = str(rep.get("contractor_ref")) == str(con["_id"]) or (con.get("email") and rep.get("contractor_email") == con.get("email"))
    if not owns:
        raise HTTPException(status_code=403, detail="This work order is not assigned to you")
    if rep.get("evidence_locked") or str(rep.get("status", "")).lower() == "resolved":
        raise HTTPException(status_code=400, detail="This work order is resolved — its evidence is locked.")
    return con, rep

@router.post("/contractor/work-orders/{report_id}/before")
async def contractor_before_photo(report_id: str, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Contractor uploads the BEFORE photo on job start → moves the work order to in_progress."""
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con, rep = await _contractor_owns_report(db, user, report_id)
    from services.cloudinary_service import upload_file_to_cloudinary
    contents = await file.read()
    result = await upload_file_to_cloudinary(contents=contents, folder="contractor_evidence")
    if not (result and result.get("url")):
        raise HTTPException(status_code=500, detail="Failed to upload before photo")
    await db["issues"].update_one(
        {"_id": rep["_id"]},
        {"$set": {
            "contractor_before_url": result["url"],
            "status": "in_progress",
            "job_started_at": datetime.utcnow(),
            "last_updated_at": datetime.utcnow().isoformat(),
        }},
    )
    return {"success": True, "before_photo_url": result["url"], "status": "in_progress"}

@router.post("/contractor/work-orders/{report_id}/after")
async def contractor_after_photo(report_id: str, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Contractor uploads the AFTER photo on completion. Reuses the existing
    verification semantics (status → pending_verification) so Staff / Dept Head
    review the before/after evidence before the report is resolved.
    """
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con, rep = await _contractor_owns_report(db, user, report_id)
    # The "before" is the citizen's report photo; the contractor just uploads the
    # completion photo, which goes to staff for verification.
    from services.cloudinary_service import upload_file_to_cloudinary
    contents = await file.read()
    result = await upload_file_to_cloudinary(contents=contents, folder="contractor_evidence")
    if not (result and result.get("url")):
        raise HTTPException(status_code=500, detail="Failed to upload after photo")
    await db["issues"].update_one(
        {"_id": rep["_id"]},
        {"$set": {
            "resolution_media_url": result["url"],
            "status": "pending_verification",
            "resolved_by": con.get("company"),
            "job_completed_at": datetime.utcnow(),
            "last_updated_at": datetime.utcnow().isoformat(),
        }},
    )
    return {"success": True, "after_photo_url": result["url"], "status": "pending_verification"}

# --- Contractor: accept work order + submit quote ---

class QuoteSubmit(BaseModel):
    amount: float
    note: Optional[str] = None

@router.post("/contractor/work-orders/{report_id}/accept")
async def contractor_accept_work_order(report_id: str, user: dict = Depends(get_current_user)):
    """Contractor accepts an assigned work order before starting the job."""
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con, rep = await _contractor_owns_report(db, user, report_id)
    await db["issues"].update_one(
        {"_id": rep["_id"]},
        {"$set": {"contractor_accepted": True, "contractor_accepted_at": datetime.utcnow(), "last_updated_at": datetime.utcnow().isoformat()}},
    )
    return {"success": True, "accepted": True}

@router.post("/contractor/work-orders/{report_id}/quote")
async def contractor_submit_quote(report_id: str, quote: QuoteSubmit, user: dict = Depends(get_current_user)):
    """Contractor submits a price quote for a work order (when the city requires one)."""
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con, rep = await _contractor_owns_report(db, user, report_id)
    q = {
        "amount": quote.amount,
        "note": quote.note,
        "submitted_at": datetime.utcnow(),
        "status": "submitted",
        "by": con.get("company"),
    }
    await db["issues"].update_one({"_id": rep["_id"]}, {"$set": {"contractor_quote": q}})
    return {"success": True, "quote": {"amount": quote.amount, "note": quote.note, "status": "submitted"}}

# --- Compliance Gate: contractor uploads docs, Dept Head reviews (pass/fail) ---

COMPLIANCE_DOCS = ["insurance", "business_license", "w9", "bonding"]

class ComplianceReview(BaseModel):
    decision: str            # "pass" | "fail"
    note: Optional[str] = None

def _compliance_state(con):
    comp = con.get("compliance") or {}
    uploaded = [d for d in COMPLIANCE_DOCS if (comp.get(d) or {}).get("url")]
    status = con.get("compliance_status")
    if not status:
        status = "pending_review" if len(uploaded) == len(COMPLIANCE_DOCS) else "incomplete"
    return {
        "compliance_status": status,
        "compliance_note": con.get("compliance_note"),
        "required": COMPLIANCE_DOCS,
        "uploaded": uploaded,
        "docs": {d: (comp.get(d) or {}) for d in COMPLIANCE_DOCS},
    }

@router.get("/contractor/compliance")
async def contractor_get_compliance(user: dict = Depends(get_current_user)):
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    db = await get_db()
    con = await db["gov_contractors"].find_one({"_id": ObjectId(user.get("id"))})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")
    return _compliance_state(con)

@router.post("/contractor/compliance/{doc_type}")
async def contractor_upload_compliance(doc_type: str, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Contractor uploads one compliance document (insurance / business_license / w9 / bonding)."""
    if user.get("type") != "contractor":
        raise HTTPException(status_code=403, detail="Contractors only")
    if doc_type not in COMPLIANCE_DOCS:
        raise HTTPException(status_code=400, detail=f"Invalid document type. Expected one of {COMPLIANCE_DOCS}")
    db = await get_db()
    con = await db["gov_contractors"].find_one({"_id": ObjectId(user.get("id"))})
    if not con:
        raise HTTPException(status_code=404, detail="Contractor not found")

    from services.cloudinary_service import upload_file_to_cloudinary
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    is_pdf = filename.endswith(".pdf") or "pdf" in content_type
    contents = await file.read()
    # PDFs upload with resource_type "auto" so Cloudinary delivers the real file;
    # images keep the default image pipeline (thumbnails, etc).
    result = await upload_file_to_cloudinary(
        contents=contents,
        folder="contractor_compliance",
        resource_type="auto" if is_pdf else "image",
    )
    if not (result and result.get("url")):
        raise HTTPException(status_code=500, detail="Failed to upload document")

    comp = con.get("compliance") or {}
    comp[doc_type] = {
        "url": result["url"],
        "status": "pending",
        "uploaded_at": datetime.utcnow(),
        "kind": "pdf" if is_pdf else "image",
        "filename": file.filename,
    }
    all_present = all((comp.get(d) or {}).get("url") for d in COMPLIANCE_DOCS)
    # Any new/changed doc returns the packet to review.
    new_status = "pending_review" if all_present else "incomplete"
    await db["gov_contractors"].update_one(
        {"_id": con["_id"]},
        {"$set": {"compliance": comp, "compliance_status": new_status}},
    )
    return {"success": True, "doc_type": doc_type, "url": result["url"], "compliance_status": new_status}

@router.post("/contractors/{contractor_id}/compliance/review")
async def review_contractor_compliance(
    contractor_id: str,
    review: ComplianceReview,
    admin: dict = Depends(require_permission("manage_authorities"))
):
    """Dept Head / Operation Manager / City Manager passes or fails a contractor's compliance packet."""
    role = str(admin.get("role", "")).lower()
    is_super = admin.get("type") in ["admin", "access"] or (admin.get("type") == "gov_portal" and role in ("super_admin", "it_super_admin"))
    is_ops_manager = admin.get("type") == "gov_portal" and role == "ops_manager"
    is_dept_admin = admin.get("type") == "gov_portal" and role == "department_admin"
    if not (is_super or is_ops_manager or is_dept_admin):
        raise HTTPException(status_code=403, detail="Not authorized to review compliance")

    db = await get_db()
    con = await _load_contractor_for_decision(db, contractor_id, admin, is_dept_admin)
    passed = str(review.decision).lower() in ("pass", "passed", "approve", "approved", "ok")
    new_status = "passed" if passed else "failed"

    comp = con.get("compliance") or {}
    for d in COMPLIANCE_DOCS:
        if comp.get(d):
            comp[d]["status"] = "approved" if passed else "rejected"

    await db["gov_contractors"].update_one(
        {"_id": con["_id"]},
        {"$set": {
            "compliance": comp,
            "compliance_status": new_status,
            "compliance_note": review.note,
            "compliance_reviewed_by": admin.get("email"),
            "compliance_reviewed_at": datetime.utcnow(),
        }},
    )
    return {"success": True, "compliance_status": new_status}

# ─────────────────────────────────────────────────────────────────────────────
# Routing Config Module — City Manager (Super Admin) only. Stores, per city, the
# recipient registry (internal + external), severity routing rules, location
# override rules, Tier-0 safety floors (locked), and an audit changelog.
# ─────────────────────────────────────────────────────────────────────────────

class RoutingRecipient(BaseModel):
    name: str
    type: str = "external"          # internal | external
    email: Optional[str] = None
    category: Optional[str] = None  # which issue category / role this recipient handles

class RoutingLocationRule(BaseModel):
    keyword: str                    # address/location keyword that triggers the override
    add_recipient: str              # recipient to add when the keyword matches
    note: Optional[str] = None

class RoutingSeverityUpdate(BaseModel):
    rules: List[dict] = []          # [{id?, severity, action}]

class RoutingCategoryRoute(BaseModel):
    issue_type: str                 # e.g. "pothole", "downed_power_line"
    route_to: str                   # department or recipient that should handle it
    note: Optional[str] = None

def _default_routing_config(city):
    now = datetime.utcnow()
    return {
        "city": city,
        "recipients": [
            {"id": str(ObjectId()), "name": "City Operations Queue", "type": "internal", "email": None,
             "category": "Global fallback / ops queue", "locked": True},
        ],
        "severity_rules": [
            {"id": str(ObjectId()), "severity": "emergency", "action": "Notify Tier-0 safety floor + assigned department immediately"},
            {"id": str(ObjectId()), "severity": "high", "action": "Notify department + Operation Manager"},
            {"id": str(ObjectId()), "severity": "medium", "action": "Route to the department queue (standard SLA)"},
            {"id": str(ObjectId()), "severity": "low", "action": "Route to the department queue (low priority)"},
        ],
        "location_overrides": [],
        # Issue category table — maps each issue type to the department / recipient
        # that should receive it. Seeded empty; the City Manager fills it (and the
        # "Suggest from departments" button on the UI pre-fills from dept issue_types).
        "category_routes": [],
        # Locked safety minimum — these can never be removed (Tier 0 floors).
        "tier0_floors": [
            {"name": "911 / Emergency Dispatch", "locked": True},
            {"name": "Police", "locked": True},
            {"name": "Fire & Emergency", "locked": True},
            {"name": "County EMA", "locked": True},
        ],
        "changelog": [{"action": "initialized", "detail": "Default routing config seeded", "by": "system", "at": now.isoformat()}],
        "created_at": now,
        "updated_at": now,
    }

async def _get_or_seed_routing(db, city):
    cfg = await db["routing_config"].find_one({"city": city})
    if not cfg:
        await db["routing_config"].insert_one(_default_routing_config(city))
        cfg = await db["routing_config"].find_one({"city": city})
    return cfg

async def _routing_log(db, city, action, detail, by):
    await db["routing_config"].update_one(
        {"city": city},
        {
            "$push": {"changelog": {"$each": [{"action": action, "detail": detail, "by": by, "at": datetime.utcnow().isoformat()}], "$slice": -100}},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )

@router.get("/routing-config")
async def get_routing_config(admin: dict = Depends(require_permission("view_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    if not city:
        raise HTTPException(status_code=400, detail="No city associated with this account")
    cfg = await _get_or_seed_routing(db, city)
    cfg["id"] = str(cfg["_id"])
    del cfg["_id"]
    cfg["changelog"] = list(reversed(cfg.get("changelog", [])))[:50]  # newest first
    cfg.setdefault("category_routes", [])  # back-fill for configs seeded before this field

    # Helper data for the Issue Category Table UI: the city's departments and the
    # issue types each one owns — so the City Manager can map type → department
    # from dropdowns instead of free-typing (and "Suggest" can pre-fill).
    depts = await db["gov_departments"].find({"city": city}).to_list(length=200)
    cfg["departments"] = [
        {"name": d.get("name"), "issue_types": d.get("issue_types") or []}
        for d in depts if d.get("name")
    ]
    return cfg

@router.post("/routing-config/recipient")
async def add_routing_recipient(rec: RoutingRecipient, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    await _get_or_seed_routing(db, city)
    item = {"id": str(ObjectId()), "name": rec.name, "type": (rec.type or "external"), "email": rec.email, "category": rec.category, "locked": False}
    await db["routing_config"].update_one({"city": city}, {"$push": {"recipients": item}})
    await _routing_log(db, city, "recipient_added", f"{item['name']} ({item['type']})", admin.get("email"))
    await record_audit(admin, "routing_recipient_added", target=item["name"], detail=f"{item['name']} ({item['type']})", city=city)
    return {"success": True, "recipient": item}

@router.delete("/routing-config/recipient/{rid}")
async def remove_routing_recipient(rid: str, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    cfg = await _get_or_seed_routing(db, city)
    target = next((r for r in cfg.get("recipients", []) if r.get("id") == rid), None)
    if target and target.get("locked"):
        raise HTTPException(status_code=400, detail="This recipient is a locked safety floor and can't be removed.")
    await db["routing_config"].update_one({"city": city}, {"$pull": {"recipients": {"id": rid}}})
    await _routing_log(db, city, "recipient_removed", (target or {}).get("name", rid), admin.get("email"))
    await record_audit(admin, "routing_recipient_removed", target=(target or {}).get("name", rid), city=city)
    return {"success": True}

@router.post("/routing-config/location-rule")
async def add_location_rule(rule: RoutingLocationRule, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    await _get_or_seed_routing(db, city)
    item = {"id": str(ObjectId()), "keyword": rule.keyword, "add_recipient": rule.add_recipient, "note": rule.note}
    await db["routing_config"].update_one({"city": city}, {"$push": {"location_overrides": item}})
    await _routing_log(db, city, "location_rule_added", f"'{rule.keyword}' -> {rule.add_recipient}", admin.get("email"))
    await record_audit(admin, "routing_location_rule_added", target=rule.keyword, detail=f"'{rule.keyword}' -> {rule.add_recipient}", city=city)
    return {"success": True, "rule": item}

@router.delete("/routing-config/location-rule/{rid}")
async def remove_location_rule(rid: str, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    await db["routing_config"].update_one({"city": city}, {"$pull": {"location_overrides": {"id": rid}}})
    await _routing_log(db, city, "location_rule_removed", rid, admin.get("email"))
    await record_audit(admin, "routing_location_rule_removed", target=rid, city=city)
    return {"success": True}

@router.put("/routing-config/severity")
async def update_severity_rules(payload: RoutingSeverityUpdate, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    await _get_or_seed_routing(db, city)
    rules = [{"id": r.get("id") or str(ObjectId()), "severity": r.get("severity"), "action": r.get("action")} for r in payload.rules]
    await db["routing_config"].update_one({"city": city}, {"$set": {"severity_rules": rules}})
    await _routing_log(db, city, "severity_rules_updated", f"{len(rules)} rules", admin.get("email"))
    await record_audit(admin, "routing_severity_updated", target=city, detail=f"{len(rules)} severity rules updated", city=city)
    return {"success": True, "severity_rules": rules}

# ── Issue Category Table — maps issue_type → handling department / recipient ──
@router.post("/routing-config/category-route")
async def add_category_route(route: RoutingCategoryRoute, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    await _get_or_seed_routing(db, city)
    issue_type = route.issue_type.strip().lower().replace(" ", "_")
    item = {"id": str(ObjectId()), "issue_type": issue_type, "route_to": route.route_to.strip(), "note": route.note}
    # Replace any existing route for the same issue type (one route per type).
    await db["routing_config"].update_one({"city": city}, {"$pull": {"category_routes": {"issue_type": issue_type}}})
    await db["routing_config"].update_one({"city": city}, {"$push": {"category_routes": item}})
    await _routing_log(db, city, "category_route_set", f"{issue_type} -> {item['route_to']}", admin.get("email"))
    await record_audit(admin, "routing_category_set", target=issue_type, detail=f"{issue_type} -> {item['route_to']}", city=city)
    return {"success": True, "route": item}

@router.delete("/routing-config/category-route/{rid}")
async def remove_category_route(rid: str, admin: dict = Depends(require_permission("manage_routing_config"))):
    db = await get_db()
    city = admin.get("org")
    cfg = await _get_or_seed_routing(db, city)
    target = next((r for r in cfg.get("category_routes", []) if r.get("id") == rid), None)
    await db["routing_config"].update_one({"city": city}, {"$pull": {"category_routes": {"id": rid}}})
    await _routing_log(db, city, "category_route_removed", (target or {}).get("issue_type", rid), admin.get("email"))
    await record_audit(admin, "routing_category_removed", target=(target or {}).get("issue_type", rid), city=city)
    return {"success": True}

# ─────────────────────────────────────────────────────────────────────────────
# Audit Log — IT Super Admin (SYSTEM) reviews who-did-what across the platform.
# Every security-sensitive admin action records one immutable entry (see
# core/audit.record_audit). A city Super Admin sees only their own city; a
# system admin sees everything.
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/audit-log")
async def get_audit_log(
    action: Optional[str] = None,
    limit: int = 100,
    admin: dict = Depends(require_permission("view_audit")),
):
    db = await get_db()
    is_system = admin.get("type") in ["admin", "access"] or str(admin.get("role", "")).lower() == "it_super_admin"
    q = {}
    # IT Super Admin (SYSTEM) sees the whole platform; a city super_admin is scoped.
    if not is_system and admin.get("org"):
        q["city"] = admin.get("org")
    if action:
        q["action"] = action
    limit = max(1, min(int(limit or 100), 500))
    docs = await db["audit_log"].find(q).sort("at", -1).to_list(length=limit)
    entries = [{
        "id": str(d.get("_id")),
        "action": d.get("action"),
        "actor_email": d.get("actor_email"),
        "actor_role": d.get("actor_role"),
        "actor_name": d.get("actor_name"),
        "target": d.get("target"),
        "detail": d.get("detail"),
        "city": d.get("city"),
        "at": d.get("at").isoformat() if hasattr(d.get("at"), "isoformat") else d.get("at"),
    } for d in docs]
    return {"entries": entries, "count": len(entries)}

@router.get("/routing-decisions")
async def get_routing_decisions(
    limit: int = 100,
    admin: dict = Depends(require_permission("view_audit")),
):
    """Auditable log of how reports were routed (recipients + AI confidence +
    whether the city's routing_config was applied). City-scoped; system sees all."""
    db = await get_db()
    is_system = admin.get("type") in ["admin", "access"] or str(admin.get("role", "")).lower() == "it_super_admin"
    q = {}
    if not is_system and admin.get("org"):
        q["city"] = admin.get("org")
    limit = max(1, min(int(limit or 100), 500))
    docs = await db["routing_decisions"].find(q).sort("created_at", -1).to_list(length=limit)
    out = [{
        "id": str(d.get("_id")),
        "issue_id": d.get("issue_id"),
        "zip_code": d.get("zip_code"),
        "city": d.get("city"),
        "issues": d.get("issues", []),
        "recipients": d.get("recipients", []),
        "has_emergency": d.get("has_emergency"),
        "city_config_applied": d.get("city_config_applied"),
        "city_config_added": d.get("city_config_added", 0),
        "at": d.get("created_at").isoformat() if hasattr(d.get("created_at"), "isoformat") else d.get("created_at"),
    } for d in docs]
    return {"entries": out, "count": len(out)}

# ─────────────────────────────────────────────────────────────────────────────
# Create City Tenant — IT Super Admin (SYSTEM) only. Spins up a new city org and
# provisions its initial City Manager (Operation Manager) with a temp password +
# branded welcome email. This is the admin-driven counterpart to the self-serve
# billing checkout flow.
# ─────────────────────────────────────────────────────────────────────────────
class CreateCityTenant(BaseModel):
    city_name: str
    state: Optional[str] = None
    primary_zip: str
    additional_zips: List[str] = []
    manager_name: str
    manager_email: EmailStr
    tier_id: Optional[str] = "spark"

@router.post("/admin/create-city")
async def create_city_tenant(
    payload: CreateCityTenant,
    admin: dict = Depends(require_permission("create_city_tenant")),
):
    import re
    db = await get_db()

    zip5 = str(payload.primary_zip).strip()
    if not (zip5.isdigit() and len(zip5) == 5):
        raise HTTPException(status_code=400, detail="Primary city ZIP must be 5 digits")
    extra = [str(z).strip() for z in (payload.additional_zips or []) if str(z).strip()]
    bad = [z for z in extra if not (z.isdigit() and len(z) == 5)]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid additional ZIP codes: {', '.join(bad)}")
    seen, all_zips = set(), []
    for z in [zip5] + extra:
        if z not in seen:
            seen.add(z); all_zips.append(z)

    city_name = payload.city_name.strip()
    if not city_name:
        raise HTTPException(status_code=400, detail="City name is required")

    # Guard against creating a duplicate city.
    if await db["organizations"].find_one({"legal_name": city_name}):
        raise HTTPException(status_code=400, detail=f"A city tenant named '{city_name}' already exists")

    # Unique slug.
    base = re.sub(r"[^a-z0-9]+", "-", f"{city_name} {payload.state or ''}".lower()).strip("-") or "city"
    slug, n = base, 1
    while await db["organizations"].find_one({"slug": slug}):
        n += 1; slug = f"{base}-{n}"

    now = datetime.utcnow()
    org_doc = {
        "slug": slug,
        "legal_name": city_name,
        "state": (payload.state or "").upper(),
        "primary_zip": zip5,
        "zip_codes": all_zips,
        "tier_id": payload.tier_id or "spark",
        "status": "active",
        "onboarding_complete": False,
        "created_at": now,
        "updated_at": now,
        "created_by": admin.get("email"),
        "provisioned_by_admin": True,
    }
    org_result = await db["organizations"].insert_one(org_doc)
    org_id = str(org_result.inserted_id)

    # Provision the initial City Manager (Operation Manager) account.
    temp_password = secrets.token_urlsafe(10)
    manager = {
        "name": payload.manager_name.strip(),
        "email": payload.manager_email.lower(),
        "department": "CITY_MANAGEMENT",
        "zip_code": zip5,
        "city": city_name,
        "org_id": org_id,
        "org_slug": slug,
        "hashed_password": get_password_hash(temp_password),
        "role": "ops_manager",
        "created_at": now,
        "created_by": admin.get("email"),
        "is_active": True,
        "require_password_change": True,
    }
    await db["government_users"].insert_one(manager)

    email_sent = False
    try:
        from services.email_service import send_city_manager_welcome_email
        email_sent = await send_city_manager_welcome_email(
            email=payload.manager_email.lower(),
            name=payload.manager_name.strip(),
            city=city_name,
            zip_code=zip5,
            temporary_password=temp_password,
            provisioned_by=admin.get("name") or admin.get("email") or "EAiSER Global Operations",
        )
    except Exception as e:
        logger.error(f"❌ City Manager welcome email failed for {payload.manager_email}: {e}", exc_info=True)

    await record_audit(
        admin, "city_tenant_created", target=city_name,
        detail=f"New city '{city_name}' ({slug}) · manager {payload.manager_email.lower()}",
        city=city_name, meta={"org_id": org_id, "slug": slug, "zips": all_zips},
    )

    return {
        "success": True,
        "org_id": org_id,
        "org_slug": slug,
        "city": city_name,
        "manager_email": payload.manager_email.lower(),
        "temp_password": temp_password,
        "email_sent": email_sent,
        "message": (
            f"City '{city_name}' created and City Manager invited."
            if email_sent else
            f"City '{city_name}' created. Welcome email could not be delivered — share the temporary password manually."
        ),
    }
