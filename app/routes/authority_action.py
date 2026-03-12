
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from jose import jwt, JWTError
from bson.objectid import ObjectId
from services.mongodb_service import get_db
from services.email_service import send_email
import os
import logging
import shutil
from pathlib import Path

router = APIRouter()
logger = logging.getLogger(__name__)

from utils.security import SECRET_KEY, ALGORITHM
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://www.eaiser.ai") # Default to live

# Re-use user auth from issues router if needed, or define here
from routes.issues_optimized_v2 import get_current_user

class UserFeedback(BaseModel):
    status: str # 'resolved' or 'persistent'

async def notify_user_status_change(user_email: str, issue_id: str, new_status: str, notes: str = None):
    """
    Notify user about the status update.
    """
    try:
        subject = f"Update on your Report #{issue_id[-6:]}"
        
        status_display = new_status.replace("_", " ").title()
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #2563eb;">Issue Status Updated</h2>
            <p>Your report has been updated by the relevant authority.</p>
            
            <div style="background: #f3f4f6; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Status:</strong> {status_display}</p>
                {f'<p><strong>Official Notes:</strong> {notes}</p>' if notes else ''}
            </div>
            
            <p>Thank you for helping improve our community.</p>
            <p style="font-size: 12px; color: #666;">EAiSER AI Team</p>
        </div>
        """
        
        await send_email(user_email, subject, html_content, f"Status updated to {status_display}")
        logger.info(f"📧 Notification sent to {user_email} for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

class AuthorityUpdate(BaseModel):
    status: str
    notes: Optional[str] = None

def create_authority_token(issue_id: str, authority_name: str = "Official Authority", days: int = 7) -> str:
    """Generate a time-limited token for authority action"""
    expire = datetime.utcnow() + timedelta(days=days)
    to_encode = {
        "sub": issue_id, 
        "scope": "authority_access", 
        "exp": expire,
        "authority_name": authority_name
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.get("/validate/{token}")
async def validate_token_and_get_issue(token: str):
    """
    Validate the token and return limited issue details (No PII).
    Used by the authority page to render the issue.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        issue_id = payload.get("sub")
        scope = payload.get("scope")
        
        if scope != "authority_access":
            raise HTTPException(status_code=403, detail="Invalid token scope")
            
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
            
        # Return only safe fields (Mask User info)
        return {
            "valid": True,
            "issue": {
                "id": issue["_id"],
                "issue_type": issue.get("issue_type"),
                "severity": issue.get("severity"),
                "description": issue.get("description"),
                "address": issue.get("address"),
                "zip_code": issue.get("zip_code"),
                "status": issue.get("status"),
                "timestamp_formatted": issue.get("timestamp_formatted"),
                "image_id": issue.get("image_id"),
                # No user_email, no personal info
            }
        }
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

@router.post("/update/{token}")
async def update_issue_via_token(token: str, update: AuthorityUpdate, background_tasks: BackgroundTasks):
    """
    Allow authority to update status/notes without login using valid token.
    """
    try:
        # 1. Verify Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        issue_id = payload.get("sub")
        
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")

        # 2. Update DB
        update_fields = {
            "status": update.status,
            "authority_notes": update.notes,
            "last_updated_by": "Authority (External)",
            "last_updated_at": datetime.utcnow().isoformat()
        }
        
        await db.issues.update_one({"_id": issue_id}, {"$set": update_fields})
        
        logger.info(f"Authority updated issue {issue_id} to {update.status}")

        # 3. Notify User (Feedback Loop)
        user_email = issue.get("user_email")
        if user_email:
            # We use the existing notification service logic
            # Passing 'notes' as context
            background_tasks.add_task(
                notify_user_status_change, 
                user_email, 
                issue_id, 
                update.status, 
                update.notes
            )

        return {"success": True, "message": "Status updated successfully"}

    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    except Exception as e:
        logger.error(f"Error updating issue via token: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

@router.get("/dashboard/{token}")
async def get_authority_dashboard(token: str):
    """
    Returns all issues for the authority associated with this token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # In a real app, the token would contain the authority_email or ID
        # For now, we'll try to find the issue linked to this token to get context,
        # or assume the token 'sub' is the authority email if it's a dashboard token.
        issue_id = payload.get("sub")
        
        db = await get_db()
        
        # If it's a single-issue token, we return that one in a list
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
             return {"valid": False, "issues": []}
             
        # Find other issues in same zip/category as a fallback list logic
        zip_code = issue.get("zip_code")
        category = issue.get("category")
        
        # Simplified: Return list of issues assigned to this 'Authority' 
        # (matching by zip and category for demo purposes)
        cursor = db.issues.find({
            "zip_code": zip_code,
            "category": category,
            "status": {"$ne": "draft"}
        }).sort("timestamp", -1).limit(20)
        
        issues = []
        async for doc in cursor:
            issues.append({
                "id": doc["_id"],
                "issue_type": doc.get("issue_type"),
                "status": doc.get("status"),
                "address": doc.get("address"),
                "severity": doc.get("severity"),
                "timestamp_formatted": doc.get("timestamp_formatted"),
                "image_id": doc.get("image_id"),
                "description": doc.get("description")
            })
            
        return {"valid": True, "issues": issues}
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return {"valid": False, "error": str(e)}

@router.post("/user-feedback/{issue_id}")
async def user_update_status_feedback(issue_id: str, feedback: UserFeedback, user: dict = Depends(get_current_user)):
    """User provides feedback on work status"""
    db = await get_db()
    
    status_map = {
        'resolved': 'resolved_by_user',
        'persistent': 'reported_persistent'
    }
    
    new_status = status_map.get(feedback.status, feedback.status)
    
    await db.issues.update_one(
        {"_id": issue_id, "user_email": user["sub"]},
        {"$set": {
            "status": new_status, 
            "user_feedback_at": datetime.utcnow().isoformat()
        }}
    )
    return {"success": True}
