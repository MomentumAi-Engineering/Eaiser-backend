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

class UserAction(BaseModel):
    user_email: str
    reason: str
    admin_id: str

# --- Endpoints ---

@router.post("/login")
async def admin_login(creds: AdminLoginRequest):
    # SIMPLE HARDCODED AUTH FOR DEMO (Replace with DB auth later)
    # Allows specific admin emails
    valid_admins = {
        "admin@eaiser.ai": "admin123",
        "chris@eaiser.ai": "admin123"
    }
    
    if creds.email in valid_admins and valid_admins[creds.email] == creds.password:
        return {
            "token": "admin-mock-token-12345", 
            "admin": {"email": creds.email, "id": "admin_001"}
        }
    raise HTTPException(status_code=401, detail="Invalid admin credentials")

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
        
        # Use optimized service method which handles connection, timeouts, and serialization
        issues = await mongo_service.get_issues_optimized(
            filter_query={"status": "needs_review"},
            skip=skip,
            limit=limit
        )
        
        # Post-process to ensure all fields needed by frontend are present
        enhanced_issues = []
        for issue in issues:
            # Flatten/Verify fields
            issue["issue_id"] = str(issue["_id"])
            if "address" not in issue:
                issue["address"] = issue.get("report", {}).get("template_fields", {}).get("location", "Unknown Location")
            
            # Ensure Image URI is set if image_id exists
            if "image_id" in issue and issue["image_id"]:
                # The frontend can use: /api/issues/{issue_id}/image
                issue["image_url"] = f"/api/issues/{issue['_id']}/image"
            
            # Ensure confidence is visible at top level
            if "confidence" not in issue:
                # Try to extract from report
                try:
                    candidates = [
                        issue.get("report", {}).get("template_fields", {}).get("confidence"),
                        issue.get("report", {}).get("unified_report", {}).get("confidence"),
                        issue.get("report", {}).get("issue_overview", {}).get("confidence"),
                    ]
                    valid = []
                    for c in candidates:
                         if c is not None:
                             s = str(c).replace('%', '').strip()
                             try: valid.append(float(s))
                             except: pass
                    if valid:
                        issue["confidence"] = min(valid) # Show min score to explain WHY it's here
                    else:
                        issue["confidence"] = 0
                except:
                    issue["confidence"] = 0

            enhanced_issues.append(issue)

        return enhanced_issues
    except Exception as e:
        logger.error(f"Failed to fetch pending reviews: {e}")
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
                
                await send_authority_email(
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
                logger.info(f"Standard Authority email triggered for approved issue {action.issue_id}")
            except Exception as e:
                logger.error(f"Failed to send authority email after approval: {e}", exc_info=True)
                # We don't rollback the approval, but log the error

        # 4. Notify User
        if issue.get("reporter_email"):
            asyncio.create_task(notify_user_status_change(issue["reporter_email"], action.issue_id, 'approved', action.notes))

        return {"message": "Issue approved and processed", "issue_id": action.issue_id}

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
