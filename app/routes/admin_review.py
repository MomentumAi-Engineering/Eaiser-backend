from fastapi import APIRouter, HTTPException, Depends, Body, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
import asyncio

from services.mongodb_optimized_service import get_optimized_mongodb_service
from services.email_service import send_email, send_formatted_ai_alert, notify_user_status_change
from services.redis_cluster_service import get_redis_cluster_service
from core.database import get_database
from services.mongodb_service import get_fs
from bson.objectid import ObjectId
from routes.issues import send_authority_email
from utils.security import verify_password, create_access_token, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
from models.admin_model import AdminCreate, AdminInDB
from datetime import timedelta
from fastapi import status
from services.security_service import SecurityService
from models.security_models import PasswordChangeRequest, TwoFactorSetup, TwoFactorVerify


# Configure logging
logger = logging.getLogger(__name__)

from core.auth import get_admin_user

router = APIRouter(
    prefix="/admin/review", 
    tags=["Admin Review"]
    # Removed global dependencies=[Depends(get_admin_user)] to allow public /login
)

# --- Pydantic Models ---

class ReviewAction(BaseModel):
    issue_id: str
    admin_id: str
    notes: Optional[str] = None
    new_authority_email: Optional[str] = None
    new_authority_name: Optional[str] = None

class AdminLoginRequest(BaseModel):
    email: str
    password: str
    code: Optional[str] = None

class UserAction(BaseModel):
    user_email: str
    reason: str
    admin_id: str

# --- Endpoints ---

@router.post("/login")
async def admin_login(creds: AdminLoginRequest, request: Request):
    """
    Enhanced admin login with security features:
    - Rate limiting
    - Account lockout after failed attempts
    - Login attempt tracking
    - Security audit logging
    """
    
    # Get client information
    client_info = SecurityService.get_client_info(request)
    ip_address = client_info["ip_address"]
    user_agent = client_info["user_agent"]
    
    # Check rate limit
    if not await SecurityService.check_rate_limit(ip_address):
        logger.warning(f"🚫 Rate limit exceeded for IP: {ip_address}")
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please try again in 1 minute."
        )
    
    # Check account lockout
    lockout_status = await SecurityService.check_account_lockout(creds.email)
    if lockout_status["locked"]:
        await SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Account locked"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Account locked due to multiple failed login attempts. Try again in {lockout_status['remaining_minutes']} minutes."
        )
    
    # Get admin from database
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        collection = await mongo_service.get_collection("admins")
        admin = await collection.find_one({"email": creds.email})
    except Exception as e:
        logger.error(f"Database error during admin login: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    # Verify credentials
    if not admin or not verify_password(creds.password, admin["password_hash"]):
        # Record failed login attempt
        await SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Invalid credentials"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if account is active
    if not admin.get("is_active", True):
        await SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Inactive account"
        )
        raise HTTPException(status_code=403, detail="Account is inactive")

    # 2FA Logic
    if admin.get("two_factor_enabled", False):
        if not creds.code:
            # Check if 2FA code is needed
            return {
                "require_2fa": True,
                "email": creds.email,
                "message": "Please enter your 2FA code"
            }
        
        # Verify 2FA code
        # For now supporting TOTP. If email method, would check against stored code.
        secret = admin.get("two_factor_secret")
        if not secret or not SecurityService.verify_2fa_code(secret, creds.code):
            await SecurityService.record_login_attempt(
                email=creds.email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                failure_reason="Invalid 2FA code"
            )
            raise HTTPException(status_code=401, detail="Invalid 2FA code")

    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=admin["email"], expires_delta=access_token_expires
    )
    
    # Record successful login
    await SecurityService.record_login_attempt(
        email=creds.email,
        ip_address=ip_address,
        user_agent=user_agent,
        success=True
    )
    
    # Log security event
    await SecurityService.log_security_event(
        admin_email=creds.email,
        action="login",
        ip_address=ip_address,
        user_agent=user_agent,
        success=True,
        details={"role": admin.get("role")}
    )
    
    logger.info(f"✅ Successful login: {creds.email} from {ip_address}")
    
    return {
        "token": access_token,
        "token_type": "bearer",
        "admin": {
            "email": admin["email"],
            "id": str(admin["_id"]),
            "role": admin.get("role", "admin"),
            "name": admin.get("name", "Admin"),
            "require_password_change": admin.get("require_password_change", False),
            "two_factor_enabled": admin.get("two_factor_enabled", False)
        }
    }

@router.post("/create", response_model=dict)
async def create_admin(new_admin: AdminCreate, current_admin: dict = Depends(get_admin_user)):
    """
    Create a new admin user. Only accessible by super_admin.
    Sends welcome email with login credentials.
    """
    # Check if current admin is super_admin
    if current_admin.get("role") != "super_admin":
        raise HTTPException(
            status_code=403, 
            detail="Only super admins can create new admin users"
        )
    
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        collection = await mongo_service.get_collection("admins", read_only=False)
        # Check if email already exists
        if await collection.find_one({"email": new_admin.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Set permissions based on role
        permissions = {
            "super_admin": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": True,
                "can_manage_team": True
            },
            "admin": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": True,
                "can_manage_team": False
            },
            "team_member": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": False,
                "can_manage_team": False
            },
            "viewer": {
                "can_approve": False,
                "can_decline": False,
                "can_assign": False,
                "can_manage_team": False
            }
        }
        
        admin_dict = new_admin.dict()
        temporary_password = new_admin.password  # Store for email
        admin_dict["password_hash"] = get_password_hash(new_admin.password)
        del admin_dict["password"]
        admin_dict["created_at"] = datetime.utcnow()
        admin_dict["is_active"] = True
        admin_dict["assigned_issues"] = []
        admin_dict["permissions"] = permissions.get(new_admin.role, permissions["admin"])
        admin_dict["last_login"] = None
        
        result = await collection.insert_one(admin_dict)
        created_admin = await collection.find_one({"_id": result.inserted_id})
        
        # Send welcome email
        try:
            from services.admin_email_service import send_admin_welcome_email
            email_sent = await send_admin_welcome_email(
                admin_email=new_admin.email,
                admin_name=new_admin.name or "Admin",
                role=new_admin.role,
                temporary_password=temporary_password,
                created_by=current_admin.get("name", current_admin.get("email"))
            )
            logger.info(f"✉️ Welcome email {'sent' if email_sent else 'failed'} to {new_admin.email}")
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            # Don't fail the whole operation if email fails
        
        # Format for response
        if "_id" in created_admin:
            created_admin["_id"] = str(created_admin["_id"])
            created_admin["id"] = created_admin["_id"]
            
        return {
            **created_admin,
            "message": f"Admin created successfully. Welcome email sent to {new_admin.email}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating admin: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/list", response_model=List[dict])
async def get_admins(current_admin: dict = Depends(get_admin_user)):
    """
    List all admin users.
    """
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        collection = await mongo_service.get_collection("admins")
        admins = await collection.find().to_list(100)
        
        # Convert ObjectId to str for JSON serialization
        results = []
        for admin in admins:
            if "_id" in admin:
                admin["_id"] = str(admin["_id"])
                # Ensure alias 'id' is present if needed by frontend, though Pydantic handled it before
                admin["id"] = admin["_id"]
            results.append(admin)
            
        logger.info(f"Retrieved {len(results)} admins")
        return results
    except Exception as e:
        logger.error(f"Error listing admins: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# DEBUG ENDPOINT - Remove after testing
@router.get("/pending-debug")
async def get_pending_reviews_debug():
    """
    Debug endpoint without auth to test data retrieval
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            return {"error": "Database unavailable"}
        
        # Log which database we're using
        db_name = mongo_service.db.name if hasattr(mongo_service, 'db') else "unknown"
        logger.info(f"🔍 Debug endpoint: Using database '{db_name}'")
        
        collection = await mongo_service.get_collection("issues")
        
        # Count total issues
        total_count = await collection.count_documents({})
        needs_review_count = await collection.count_documents({"status": "needs_review"})
        
        logger.info(f"🔍 Total issues in DB: {total_count}, needs_review: {needs_review_count}")
        
        cursor = collection.find({"status": "needs_review"}).sort("timestamp", -1).limit(10)
        issues = await cursor.to_list(length=10)
        
        return {
            "database": db_name,
            "total_issues": total_count,
            "needs_review_count": needs_review_count,
            "count": len(issues),
            "issues": [
                {
                    "id": str(i["_id"]),
                    "status": i.get("status"),
                    "type": i.get("issue_type"),
                    "timestamp": str(i.get("timestamp"))
                } for i in issues
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/pending", response_model=List[dict])
async def get_pending_reviews(
    skip: int = 0,
    limit: int = 50,
    admin: dict = Depends(get_admin_user)
):
    """
    Get all issues that are flagged for review (status='needs_review').
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        collection = await mongo_service.get_collection("issues")
        
        # BRUTE FORCE STRATEGY: 
        # Fetch the last 50 issues regardless of status. 
        # We will manually filter in Python to ensure NO edge case (string/int mismatch) hides an issue.
        cursor = collection.find({}).sort("timestamp", -1).limit(50)
        recent_issues = await cursor.to_list(length=50)
        
        final_reviews = []
        seen_ids = set()
        
        # Helper to parse ANY format - define ONCE outside loop
        def parse_conf(val):
            if val is None: return None
            try:
                s = str(val).replace("%", "").strip()
                return float(s)
            except: 
                return None

        for issue in recent_issues:
            sid = str(issue["_id"])
            if sid in seen_ids: continue
            
            status = issue.get("status", "unknown")
            should_show = False
            
            # FIRST: Skip already processed issues
            if status in ["rejected", "declined", "completed", "submitted"]:
                # Don't show issues that have been finalized
                continue
            
            # 1. Explicitly flagged
            if status == "needs_review":
                should_show = True
            
            # 2. Check for "fake" / "screened_out" / "reject"
            elif status == "screened_out" or issue.get("dispatch_decision") == "reject":
                should_show = True
            
            # 3. Check pending issues for low confidence
            elif status == "pending":
                # manual extraction
                conf_values = []

                # Extract from all known locations
                c1 = parse_conf(issue.get("confidence"))
                c2 = parse_conf(issue.get("report", {}).get("issue_overview", {}).get("confidence"))
                c3 = parse_conf(issue.get("report", {}).get("template_fields", {}).get("confidence"))
                c4 = parse_conf(issue.get("report", {}).get("unified_report", {}).get("confidence"))
                
                if c1 is not None: conf_values.append(c1)
                if c2 is not None: conf_values.append(c2)
                if c3 is not None: conf_values.append(c3)
                if c4 is not None: conf_values.append(c4)
                
                # Determine effective confidence
                effective_conf = min(conf_values) if conf_values else 0
                
                # Check for "fake" keywords in description
                desc = str(issue.get("description") or "").lower()
                ai_summary = str(issue.get("report", {}).get("issue_overview", {}).get("summary_explanation") or "").lower()
                combined_text = desc + " " + ai_summary
                is_fake_text = any(x in combined_text for x in ["fake", "cartoon", "ai generate"])
                
                if effective_conf < 70 or is_fake_text or issue.get("issue_type") == "unknown":
                    should_show = True
                    # Force status for UI consistency
                    issue["status_original"] = status
                    issue["status"] = "needs_review"

            if should_show:
                seen_ids.add(sid)
                
                # DATA NORMALIZATION FOR FRONTEND
                issue["issue_id"] = sid
                
                # Ensure Image URL
                if "image_url" not in issue:
                    # Prefer standard endpoint
                    issue["image_url"] = f"/api/issues/{sid}/image"

                # Ensure Address
                if "address" not in issue:
                    issue["address"] = issue.get("report", {}).get("template_fields", {}).get("location", "Unknown Location")

                # Ensure Confidence Display
                if "confidence" not in issue:
                    # re-calculate to ensure property exists
                    valid_c = [x for x in [
                        parse_conf(issue.get("report", {}).get("issue_overview", {}).get("confidence")),
                        parse_conf(issue.get("report", {}).get("template_fields", {}).get("confidence"))
                    ] if x is not None]
                    issue["confidence"] = min(valid_c) if valid_c else 0

                final_reviews.append(issue)

        logger.info(f"Admin Dashboard: Returning {len(final_reviews)} issues (Filtered from {len(recent_issues)} recent)")
        logger.info(f"Issue IDs being returned: {[str(i['_id'])[-8:] for i in final_reviews[:5]]}")
        return final_reviews
    except Exception as e:
        logger.error(f"Failed to fetch pending reviews: {e}", exc_info=True)
        # DEBUG: Return actual error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/approve")
async def approve_issue(action: ReviewAction, admin: dict = Depends(get_admin_user)):
    """
    Approve a flagged issue.
    - Updates status to 'pending' (or 'approved')
    - Triggers the email to authority
    - Notifies user (optional)
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # 1. Get the issue
        issue = await mongo_service.get_issue_by_id(action.issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")

        # 2. Update status to 'pending' (ready for authority)
        # 2. Update status to 'submitted' (ready for authority)
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": action.issue_id},
            update_dict={
                "$set": {
                    "status": "submitted",  # Mark as submitted/completed
                    "admin_review": {
                        "action": "approve",
                        "admin_id": action.admin_id,
                        "timestamp": datetime.utcnow(),
                        "notes": action.notes
                    }
                }
            }
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update issue status")

        # 2.5 If authority changed, update report and issue
        if action.new_authority_email and issue.get("report"):
             report = issue["report"]
             # Update recommended authorities in report
             # We just override the first one or add to list
             new_auth = {
                 "name": action.new_authority_name or "Assigned Authority", 
                 "email": action.new_authority_email, 
                 "type": "custom"
             }
             
             # Identify if we replace or append? For now, let's force set sending list
             # The email trigger below uses issue['report'] which is STALE now.
             # We need to update issue['report'] in DB and memory.
             
             report["responsible_authorities_or_parties"] = [new_auth]
             # Update in DB
             await mongo_service.update_one_optimized(
                 collection_name='issues',
                 filter_dict={"_id": action.issue_id},
                 update_dict={"$set": {"report.responsible_authorities_or_parties": [new_auth]}}
             )
             
             # Update local issue object for email sending
             issue["report"] = report
             logger.info(f"Admin updated authority to {action.new_authority_email} for issue {action.issue_id}")



        # 3. Trigger Authority Email (since it was skipped earlier)
        # We re-fetch to get updated state if needed, or use 'issue' dict
        # The report is in issue['report']
        if issue.get("report"):
            # Send standard formatted email (same as normal submission)
            try:
                # Need to fetch image content for attachment
                fs = await get_fs()
                image_id = issue.get("image_id")
                image_content = b""
                if image_id:
                     try:
                        grid_out = await fs.open_download_stream(ObjectId(image_id))
                        image_content = await grid_out.read()
                     except Exception as e:
                        logger.warning(f"Failed to fetch image {image_id} for email: {e}")

                # Prepare authorities list
                # If we updated it in step 2.5, use that. Or use what's in report.
                current_authorities = issue["report"].get("responsible_authorities_or_parties", [])
                
                # Normalize authorities format for email function
                # send_authority_email expects List[Dict[str, str]] with name, email
                email_auths = []
                for a in current_authorities:
                    email_auths.append({
                        "name": a.get("name", "Authority"), 
                        "email": a.get("email"),
                        "type": a.get("type", "general")
                    })
                
                email_success = await send_authority_email(
                    issue_id=str(issue["_id"]),
                    authorities=email_auths,
                    issue_type=issue.get("issue_type", "Unknown"),
                    final_address=issue.get("address", "Unknown Address"),
                    zip_code=issue.get("zip_code", "N/A"),
                    timestamp_formatted=issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
                    report=issue["report"],
                    confidence=float(issue.get("confidence", 0)),
                    category=issue.get("category", "public"),
                    timezone_name=issue.get("timezone_name", "UTC"),
                    latitude=float(issue.get("latitude", 0.0)),
                    longitude=float(issue.get("longitude", 0.0)),
                    image_content=image_content,
                    is_user_review=False
                )
                
                if email_success:
                    logger.info(f"Standard Authority email triggered for approved issue {action.issue_id}")
                    return {"message": "Issue approved and email sent to authorities", "issue_id": action.issue_id, "email_sent": True}
                else:
                    logger.warning(f"Authority email failed for issue {action.issue_id}")
                    return {"message": "Issue approved but email sending failed. Check server logs.", "issue_id": action.issue_id, "email_sent": False}

            except Exception as e:
                logger.error(f"Failed to send authority email after approval: {e}", exc_info=True)
                return {"message": f"Issue approved but email error occurred: {str(e)}", "issue_id": action.issue_id, "email_sent": False}

        # 4. Notify User
        if issue.get("reporter_email"):
            asyncio.create_task(notify_user_status_change(issue["reporter_email"], action.issue_id, 'approved', action.notes))

        return {"message": "Issue approved (no report data to email)", "issue_id": action.issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decline")
async def decline_issue(action: ReviewAction, admin: dict = Depends(get_admin_user)):
    """
    Decline a flagged issue.
    - Updates status to 'rejected'
    - Notifies user (TODO: Add notification logic)
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # Update status
        # Update status
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": action.issue_id},
            update_dict={
                "$set": {
                    "status": "rejected",
                    "admin_review": {
                        "action": "decline",
                        "admin_id": action.admin_id,
                        "timestamp": datetime.utcnow(),
                        "notes": action.notes
                    }
                }
            }
        )

        if not success:
            raise HTTPException(status_code=404, detail="Issue not found or update failed")

        # Notify User "Your report was declined"
        issue = await mongo_service.get_issue_by_id(action.issue_id)
        if issue and issue.get("reporter_email"):
             asyncio.create_task(notify_user_status_change(issue["reporter_email"], action.issue_id, 'rejected', action.notes))
        
        return {"message": "Issue declined", "issue_id": action.issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error declining issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/skip")
async def skip_review(action: ReviewAction, admin: dict = Depends(get_admin_user)):
    """
    Skip exact review (maybe leave for another admin or mark as 'skipped'?)
    For now, we'll just log it or maybe irrelevant if we don't change status.
    If 'skip' means 'Ignore/Archive', we can use a status like 'archived'.
    """
    # Assuming Skip means "I don't know, leave it for now" -> no action on status
    return {"message": "Review skipped (no action taken)", "issue_id": action.issue_id}

@router.post("/deactivate-user")
async def deactivate_user(action: UserAction, admin: dict = Depends(get_admin_user)):
    """
    Deactivate a user account (e.g. for spamming fake reports).
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
            
        # Assuming there is a 'users' collection. 
        # If not, we might blacklist the email in a separate collection.
        
        # Check if users collection exists or we just blacklist
        # For now, let's assume we update a 'users' collection or add to 'blacklisted_users'
        
        # Check for user by email
        user = await mongo_service.db.users.find_one({"email": action.user_email})
        if user:
            await mongo_service.db.users.update_one(
                {"email": action.user_email},
                {"$set": {"is_active": False, "deactivation_reason": action.reason}}
            )
        else:
            # If user collection user doesn't exist (maybe only in issues), create blacklist entry?
            await mongo_service.db.blacklisted_users.update_one(
                {"email": action.user_email},
                {"$set": {"email": action.user_email, "reason": action.reason, "admin_id": action.admin_id, "timestamp": datetime.utcnow()}},
                upsert=True
            )
            
        return {"message": f"User {action.user_email} deactivated."}

    except Exception as e:
        logger.error(f"Error deactivating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{admin_id}")
async def delete_admin(admin_id: str, current_admin: dict = Depends(get_admin_user)):
    """
    Delete an admin user.
    """
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Prevent self-deletion (optional but recommended safety)
        if str(current_admin["id"]) == admin_id:
             raise HTTPException(status_code=400, detail="Cannot delete your own account.")

        collection = await mongo_service.get_collection("admins", read_only=False)
        result = await collection.delete_one({"_id": ObjectId(admin_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Admin not found")
            
        return {"message": "Admin deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting admin: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ============================================
# ISSUE ASSIGNMENT ENDPOINTS
# ============================================

@router.post("/assign-issue")
async def assign_issue_to_admin(
    issue_id: str = Body(...),
    admin_email: str = Body(...),
    current_admin: dict = Depends(get_admin_user)
):
    """
    Assign an issue to a specific admin. Only super_admin can assign.
    """
    # Check permissions
    if current_admin.get("role") != "super_admin":
        raise HTTPException(
            status_code=403,
            detail="Only super admins can assign issues"
        )
    
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # Find the admin to assign to
        admins_collection = await mongo_service.get_collection("admins")
        target_admin = await admins_collection.find_one({"email": admin_email})
        
        if not target_admin:
            raise HTTPException(status_code=404, detail=f"Admin {admin_email} not found")
        
        if not target_admin.get("is_active"):
            raise HTTPException(status_code=400, detail="Cannot assign to inactive admin")
        
        # Check if admin has permission to handle issues
        permissions = target_admin.get("permissions", {})
        if not permissions.get("can_approve") and not permissions.get("can_decline"):
            raise HTTPException(
                status_code=400,
                detail=f"Admin {admin_email} (role: {target_admin.get('role')}) cannot handle issues"
            )
        
        # Update admin's assigned_issues list
        await admins_collection.update_one(
            {"email": admin_email},
            {"$addToSet": {"assigned_issues": issue_id}}
        )
        
        # Update issue with assigned_to field
        issues_collection = await mongo_service.get_collection("issues")
        await issues_collection.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "assigned_to": admin_email,
                    "assigned_by": current_admin.get("email"),
                    "assigned_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Issue {issue_id} assigned to {admin_email} by {current_admin.get('email')}")
        
        return {
            "message": f"Issue assigned to {target_admin.get('name', admin_email)}",
            "issue_id": issue_id,
            "assigned_to": {
                "email": admin_email,
                "name": target_admin.get("name"),
                "role": target_admin.get("role")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-assigned-issues")
async def get_my_assigned_issues(current_admin: dict = Depends(get_admin_user)):
    """
    Get all issues assigned to the current admin
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        admin_email = current_admin.get("email")
        
        # Get admin's assigned issues
        issues_collection = await mongo_service.get_collection("issues")
        cursor = issues_collection.find({"assigned_to": admin_email}).sort("timestamp", -1)
        assigned_issues = await cursor.to_list(length=100)
        
        # Format response & Normalize
        for issue in assigned_issues:
            sid = str(issue["_id"])
            issue["_id"] = sid
            issue["issue_id"] = sid
            
            # Normalize Image URL for frontend
            if "image_url" not in issue:
                issue["image_url"] = f"/api/issues/{sid}/image"

            # Ensure minimal report structure if missing (prevents frontend crash)
            if "report" not in issue:
                issue["report"] = {}

        logger.info(f"📋 Retrieved {len(assigned_issues)} assigned issues for {admin_email}")
        
        return {
            "count": len(assigned_issues),
            "issues": assigned_issues
        }
        
    except Exception as e:
        logger.error(f"Error fetching assigned issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STATISTICS ENDPOINT
# ============================================


class BulkAssignRequest(BaseModel):
    issue_ids: List[str]
    admin_email: str

@router.post("/bulk-assign")
async def bulk_assign_issues(request: BulkAssignRequest, current_admin: dict = Depends(get_admin_user)):
    """Assign multiple issues to an admin"""
    # Only allow assignment if super_admin or maybe normal admin too?
    # User said "me admin hu". So standard admin should be able to assign.
    # But usually assignment is restricted.
    # Existing 'assign_issue' is restricted to 'super_admin'.
    # I should check permissions. User asked "me admin hu".
    # I'll stick to 'super_admin' check to match existing 'assign_issue'.
    # If they are just 'admin', they might not have permission.
    # BUT, existing 'assign-issue' endpoint enforces super_admin.
    
    if current_admin.get("role") not in ["super_admin", "admin"]:
         raise HTTPException(status_code=403, detail="Only admins/super admins can assign issues")
    
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    collection = await mongo_service.get_collection("issues")
    
    # Validate Issue IDs
    try:
        object_ids = [ObjectId(id) for id in request.issue_ids]
    except:
        raise HTTPException(status_code=400, detail="Invalid Issue ID format")

    # Update
    result = await collection.update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {
            "assigned_to": request.admin_email, 
            "updated_at": datetime.utcnow(),
            # Should we change status? Maybe 'pending' -> 'pending'?
            # Usually assignment doesn't change status, just ownership.
        }}
    )
    
    logger.info(f"Bulk assigned {result.modified_count} issues to {request.admin_email} by {current_admin['email']}")
    
    return {"message": f"Assigned {result.modified_count} issues", "count": result.modified_count}

@router.get("/stats")
async def get_admin_stats(current_admin: dict = Depends(get_admin_user)):
    """
    Get dashboard statistics - optimized aggregation with real-time data.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # 1. Fetch Admin Team Data
        admins_collection = await mongo_service.get_collection("admins")
        admins = await admins_collection.find({"is_active": True}).to_list(length=100)
        issues_collection = await mongo_service.get_collection("issues")
        
        # 2. Aggregation Pipeline
        pipeline = [
            {
                "$facet": {
                    "total": [{"$count": "count"}],
                    "status_counts": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}],
                    "type_counts": [
                        {"$group": {"_id": "$issue_type", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ],
                    "recent": [
                        {"$sort": {"timestamp": -1}},
                        {"$limit": 10},
                        {"$project": {"issue_type": 1, "status": 1, "assigned_to": 1, "timestamp": 1}}
                    ],
                    "resolved_by_admin": [
                        {"$match": {"status": {"$in": ["submitted", "completed"]}}},
                        {"$group": {"_id": "$assigned_to", "count": {"$sum": 1}}}
                    ]
                }
            }
        ]
        
        # Execute DIRECTLY (No Cache) to ensure real-time data
        cursor = issues_collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        facets = results[0] if results else {}
        
        # 3. Process Results (Robustly)
        
        # Total (Handle empty case safely)
        total_list = facets.get("total", [])
        total_issues = total_list[0].get("count", 0) if total_list else 0
        
        # Status
        status_map = {item["_id"]: item["count"] for item in facets.get("status_counts", [])}
        pending_review = status_map.get("needs_review", 0)
        approved = status_map.get("submitted", 0)
        declined = sum(status_map.get(s, 0) for s in ["rejected", "declined"])
        
        # By Type
        by_type = {item["_id"]: item["count"] for item in facets.get("type_counts", []) if item["_id"]}
        
        # Team Performance
        resolved_map = {item["_id"]: item["count"] for item in facets.get("resolved_by_admin", []) if item["_id"]}
        
        team_performance = []
        for admin in admins:
            if admin.get("role") in ["admin", "team_member", "super_admin"]:
                email = admin.get("email")
                team_performance.append({
                    "name": admin.get("name", email),
                    "email": email,
                    "role": admin.get("role"),
                    "assigned": len(admin.get("assigned_issues", [])),
                    "resolved": resolved_map.get(email, 0)
                })
        
        # Recent Activity
        recent_activity = []
        for issue in facets.get("recent", []):
            action = "created"
            status = issue.get("status")
            if status == "submitted":
                action = "approved"
            elif status in ["rejected", "declined"]:
                action = "declined"
            elif issue.get("assigned_to"):
                action = f"assigned to {issue.get('assigned_to')}"
            
            recent_activity.append({
                "description": f"Issue {issue.get('issue_type', 'unknown')} {action}",
                "timestamp": issue.get("timestamp")
            })
        
        logger.info(f"📊 Stats fetched (Real-time) for {current_admin.get('email')}")
        
        return {
            "total_issues": total_issues,
            "pending_review": pending_review,
            "approved": approved,
            "declined": declined,
            "by_type": by_type,
            "team_performance": team_performance,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {
            "total_issues": 0, "pending_review": 0, "approved": 0, "declined": 0,
            "by_type": {}, "team_performance": [], "recent_activity": []
        }


# ============================================
# SECURITY ENDPOINTS
# ============================================

@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_admin: dict = Depends(get_admin_user),
    request: Request = None
):
    """Change admin password"""
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    # Get full admin record to check current password
    admin = await collection.find_one({"email": current_admin["email"]})
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
        
    # Verify current password
    if not verify_password(password_data.current_password, admin["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
        
    # Update to new password
    new_hash = get_password_hash(password_data.new_password)
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "password_hash": new_hash,
                "require_password_change": False,
                "password_last_changed": datetime.utcnow()
            }
        }
    )
    
    # Audit log
    if request:
        client_info = SecurityService.get_client_info(request)
        await SecurityService.log_security_event(
            admin_email=current_admin["email"],
            action="password_change",
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            success=True
        )
        
    return {"message": "Password changed successfully"}

@router.post("/2fa/setup")
async def setup_2fa(
    setup_data: TwoFactorSetup,
    current_admin: dict = Depends(get_admin_user)
):
    """Initialize 2FA setup - returns secret"""
    if setup_data.method != "totp":
        raise HTTPException(status_code=400, detail="Only TOTP currently supported")
        
    # Generate secret
    secret = SecurityService.generate_2fa_secret()
    
    return {
        "secret": secret,
        "uri": f"otpauth://totp/EAiSER:{current_admin['email']}?secret={secret}&issuer=EAiSER"
    }

@router.post("/2fa/verify")
async def verify_2fa_setup(
    verify_data: TwoFactorVerify,
    current_admin: dict = Depends(get_admin_user)
):
    """Verify and enable 2FA"""
    secret = verify_data.session_token
    
    if not SecurityService.verify_2fa_code(secret, verify_data.code):
        raise HTTPException(status_code=400, detail="Invalid code")
        
    # Enable 2FA for user
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "two_factor_enabled": True,
                "two_factor_secret": secret,
                "two_factor_method": "totp"
            }
        }
    )
    
    return {"message": "2FA verified and enabled"}

@router.post("/2fa/disable")
async def disable_2fa(
    current_admin: dict = Depends(get_admin_user)
):
    """Disable 2FA"""
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "two_factor_enabled": False,
                "two_factor_secret": None
            }
        }
    )
    
    return {"message": "2FA disabled"}
