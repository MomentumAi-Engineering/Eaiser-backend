from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
from datetime import datetime

from services.mongodb_service import get_db
from core.auth import require_permission

router = APIRouter(
    prefix="/admin/settings",
    tags=["Admin Settings"]
)

logger = logging.getLogger(__name__)

class MaintenanceToggleRequest(BaseModel):
    enabled: bool

@router.get("/maintenance-status")
async def get_maintenance_status():
    """
    Get current maintenance mode status.
    """
    try:
        db = await get_db()
        settings = await db["settings"].find_one({"key": "maintenance_mode"})
        
        if not settings:
            return {"enabled": False}
            
        return {"enabled": settings.get("value", False)}
    except Exception as e:
        logger.error(f"Error getting maintenance status: {e}")
        return {"enabled": False}

@router.post("/maintenance-toggle")
async def toggle_maintenance_mode(
    request: MaintenanceToggleRequest,
    admin: dict = Depends(require_permission("maintenance_mode"))
):
    """
    Toggle maintenance mode (Super Admin Only).
    """
    try:
        db = await get_db()
        
        await db["settings"].update_one(
            {"key": "maintenance_mode"},
            {"$set": {
                "value": request.enabled,
                "updated_by": admin.get("email"),
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        
        status = "enabled" if request.enabled else "disabled"
        logger.warning(f"⚠️ Maintenance Mode {status} by {admin.get('email')}")
        
        return {"message": f"Maintenance mode {status} successfully", "enabled": request.enabled}
    except Exception as e:
        logger.error(f"Error toggling maintenance mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))
