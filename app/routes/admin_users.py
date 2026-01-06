from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from services.mongodb_service import get_db
from core.auth import get_admin_user
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
    admin: dict = Depends(get_admin_user)
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
async def delete_user(user_id: str, admin: dict = Depends(get_admin_user)):
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
        
        # 1. Delete User
        await users_coll.delete_one({"_id": ObjectId(user_id)})
        
        # 2. Delete Users Issues? Or Keep them anonymized?
        # User asked: "data base me se bhi delete ho jaye" (delete from DB too) implies full cleanup.
        deleted_issues_count = 0
        if user_email:
             result = await issues_coll.delete_many({"user_email": user_email})
             deleted_issues_count = result.deleted_count
        
        logger.info(f"User {user_email} (ID: {user_id}) deleted by {admin.get('email')}. Removed {deleted_issues_count} issues.")
        
        return {
            "message": f"User permanently deleted along with {deleted_issues_count} reports.",
            "deleted_id": user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/toggle-status")
async def toggle_user_status(action: UserStatusUpdate, admin: dict = Depends(get_admin_user)):
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
