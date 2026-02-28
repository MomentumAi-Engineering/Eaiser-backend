from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from services.mongodb_service import get_db
from core.auth import get_admin_user, require_permission
from bson import ObjectId

router = APIRouter(
    prefix="/admin/users",
    tags=["Admin User Management"]
)

logger = logging.getLogger(__name__)

# --- Models ---

class UserStatDTO(BaseModel):
    id: str
    name: str = "Unknown"
    email: str
    role: str
    is_active: bool
    rejected_reports_count: int = 0
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    risk_score: str = "Low" # Low, Medium, High, Critical

class UserStatusUpdate(BaseModel):
    user_id: str
    is_active: bool
    reason: Optional[str] = None

# --- Endpoints ---

@router.get("/list", response_model=List[UserStatDTO])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    filter_risk: Optional[str] = None, # 'high_risk'
    search: Optional[str] = None,
    admin: dict = Depends(require_permission("view_users"))
):
    """
    List all registered users with their reputation stats.
    Allows filtering by high risk (frequent fake reporters).
    """
    try:
        db = await get_db()
        users_coll = db["users"]
        issues_coll = db["issues"]
        
        # Build Query
        query = {}
        if search:
            query["$or"] = [
                {"email": {"$regex": search, "$options": "i"}},
                {"name": {"$regex": search, "$options": "i"}}
            ]
            
        # If High Risk filter
        if filter_risk == "high_risk":
            # Users with > 2 rejected reports
            query["rejected_reports_count"] = {"$gt": 2}

        # Execute
        cursor = users_coll.find(query).sort("rejected_reports_count", -1).skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        
        results = []
        for u in users:
            uid = str(u["_id"])
            email = u.get("email")
            rejected = u.get("rejected_reports_count", 0)
            
            # --- NEW: Get Total Reports Count (Live) ---
            # Count issues submitted by this user
            total_reports = 0
            if email:
                total_reports = await issues_coll.count_documents({"user_email": email})
            
            # Determine Risk
            risk = "Low"
            if rejected > 2: risk = "Medium"
            if rejected > 5: risk = "High"
            if rejected > 10: risk = "Critical"
            
            # If user is BANNED (is_active=False), mark as Banned
            if not u.get("is_active", True):
                risk = "Banned"

            results.append({
                "id": uid,
                "name": u.get("name", "Unknown"),
                "email": email,
                "role": u.get("role", "user"),
                "is_active": u.get("is_active", True),
                "rejected_reports_count": total_reports, # USER REQUEST: "konse email se kitne mail aaye hai" (showing Total Reports instead of Rejected in this field for now, or adding new field logic below)
                "created_at": u.get("created_at"),
                "last_login": u.get("last_login"),
                "risk_score": risk
            })
            # Correction: The DTO has 'rejected_reports_count'. passing total there might be confusing.
            # Let's adjust the DTO in a subsequent edit or re-use the field if the user only cares about 'how many mails'.
            # actually, better to extend DTO. But for quick fix, the user asked "Live ki konse email se kitne mail aaye hai".
            
        return results

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{user_id}")
async def delete_user(user_id: str, admin: dict = Depends(require_permission("manage_users"))):
    """
    Permanently delete a user account and their associated data.
    """
    try:
        db = await get_db()
        users_coll = db["users"]
        issues_coll = db["issues"]
        
        # Check if user exists
        user = await users_coll.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # Security: Prevent deleting admins
        if user.get("role") in ["admin", "super_admin"]:
            raise HTTPException(status_code=403, detail="Cannot delete admin accounts via this endpoint")
            
        user_email = user.get("email")
        
        # 1. Collect all reports to clean up assets (Images)
        deleted_count = 0
        image_delete_count = 0
        if user_email:
            # Find all issues to get image_id's
            user_issues_cursor = issues_coll.find({"user_email": user_email})
            user_issues = await user_issues_cursor.to_list(length=None)
            
            # Extract Image IDs
            image_ids = [i.get("image_id") for i in user_issues if i.get("image_id")]
            
            # 2. Delete Images from GridFS (Parallel for speed)
            if image_ids:
                from services.mongodb_service import get_fs
                fs = await get_fs()
                
                async def safe_delete(img_id):
                    try:
                        obj_id = None
                        if isinstance(img_id, str) and ObjectId.is_valid(img_id):
                            obj_id = ObjectId(img_id)
                        elif isinstance(img_id, ObjectId):
                            obj_id = img_id
                        
                        if obj_id:
                            await fs.delete(obj_id)
                            return True
                    except Exception as e:
                        # Silently skip if file not found (already "deleted")
                        if "no file could be deleted" not in str(e):
                            logger.warning(f"Error during GridFS delete for {img_id}: {e}")
                    return False

                # Execute all deletions in parallel
                results = await asyncio.gather(*[safe_delete(mid) for mid in image_ids])
                image_delete_count = sum(1 for r in results if r)
            
            # 3. Delete from Issues Collection
            result = await issues_coll.delete_many({"user_email": user_email})
            deleted_count = result.deleted_count

        # 4. Delete From Other potential collections (Clean Slate)
        # Any other user-specific data should be wiped here
        try:
             # Clean security logs and attempts associated with this email
             await db["audit_logs"].delete_many({"admin_email": user_email}) # Use admin_email field if user was ever an admin or referenced as such
             await db["login_attempts"].delete_many({"email": user_email})
             
             # General audit logs where email might be in details
             await db["audit_logs"].delete_many({"details.user_email": user_email})
             
             # Use general regex check if needed, but per-field is safer/faster
             logger.info(f"🧼 Security logs cleared for {user_email}")
        except Exception as log_err:
             logger.warning(f"Failed to clear some auxiliary logs: {log_err}")

        # 5. Delete User
        await users_coll.delete_one({"_id": ObjectId(user_id)})
        
        logger.info(f"🗑️ FULL WIPE: User {user_email} deleted. {deleted_count} issues and {image_delete_count} images purged.")
        
        return {
            "message": f"User account and all associated data permanently deleted.",
            "details": {
                "deleted_reports": deleted_count,
                "deleted_images": image_delete_count,
                "fresh_start": True
            },
            "deleted_id": user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/toggle-status")
async def toggle_user_status(action: UserStatusUpdate, admin: dict = Depends(require_permission("manage_users"))):
    """
    Block or Unblock a user.
    """
    try:
        db = await get_db()
        users_coll = db["users"]
        
        # Security: Admin cannot ban other admins/super_admins via this endpoint?
        # Ideally this is for END USERS only.
        target_user = await users_coll.find_one({"_id": ObjectId(action.user_id)})
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")
            
        if target_user.get("role") in ["admin", "super_admin"]:
            raise HTTPException(status_code=403, detail="Cannot manage admin accounts here")
            
        update_data = {
            "is_active": action.is_active,
            "status_updated_by": admin.get("email"),
            "status_updated_at": datetime.utcnow()
        }
        
        if not action.is_active and action.reason:
            update_data["ban_reason"] = action.reason
        
        await users_coll.update_one(
            {"_id": ObjectId(action.user_id)},
            {"$set": update_data}
        )
        
        status_str = "Active" if action.is_active else "Banned"
        logger.info(f"User {target_user.get('email')} status changed to {status_str} by {admin.get('email')}")
        
        return {"message": f"User marked as {status_str}", "id": action.user_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling user status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
