import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.services.mongodb_service import get_db

logger = logging.getLogger(__name__)

# Global In-Memory Cache
ISSUE_DEPARTMENT_MAP = {}
ISSUE_CATEGORY_MAP = {}
ZIP_CODE_AUTHORITIES = {}

# AUDIT LOGGING
async def log_audit_event(
    action: str, 
    target: str, 
    details: Dict[str, Any], 
    admin_email: str = "system"
):
    """
    Log an administrative action to the database.
    
    Args:
        action: "create_mapping", "update_mapping", "resolve_review", "update_zip"
        target: The entity being modified (e.g., issue_type name or zip code)
        details: What changed (old_value, new_value, etc.)
        admin_email: Who performed the action
    """
    try:
        db = await get_db()
        entry = {
            "action": action,
            "target": target,
            "details": details,
            "admin_email": admin_email,
            "timestamp": datetime.utcnow().isoformat(),
            "id": str(uuid.uuid4())
        }
        await db.audit_logs.insert_one(entry)
        logger.info(f"📝 Audit Logged: {action} on {target} by {admin_email}")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

def load_mappings(base_path: Optional[str] = None):
    """Load JSON mappings from the data directory."""
    global ISSUE_DEPARTMENT_MAP, ISSUE_CATEGORY_MAP, ZIP_CODE_AUTHORITIES
    
    if base_path is None:
        # Default to app/data
        # Assuming this file is in app/services/, parent is app, parent.parent is root (Eaiser-backend) ?
        # Actually Path(__file__).parent is app/services
        # Path(__file__).parent.parent is app
        # Path(__file__).parent.parent / "data" is app/data
        base_path = Path(__file__).parent.parent / "data"
    else:
        base_path = Path(base_path)

    try:
        with open(base_path / 'issue_department_map.json', 'r') as f:
            ISSUE_DEPARTMENT_MAP = json.load(f)
        logger.info("✅ Loaded issue_department_map.json")
    except Exception as e:
        logger.error(f"❌ Failed to load issue_department_map.json: {e}")

    try:
        with open(base_path / 'issue_category_map.json', 'r') as f:
            ISSUE_CATEGORY_MAP = json.load(f)
        logger.info("✅ Loaded issue_category_map.json")
    except Exception as e:
        logger.error(f"❌ Failed to load issue_category_map.json: {e}")

    try:
        with open(base_path / 'zip_code_authorities.json', 'r') as f:
            ZIP_CODE_AUTHORITIES = json.load(f)
        logger.info("✅ Loaded zip_code_authorities.json")
    except Exception as e:
        logger.error(f"❌ Failed to load zip_code_authorities.json: {e}")

async def resolve_authorities(issue_type: str, zip_code: str, ai_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    3-tier authority resolution:
    1. Try exact issue type mapping
    2. Fall back to category
    3. Flag unmapped & route to general
    """
    authorities = []
    mapping_review_entry = None
    is_mapped = True
    
    # Clean inputs
    issue_type_key = issue_type.lower().strip().replace(' ', '_')
    
    # TIER 1: Check direct issue type mapping
    departments = ISSUE_DEPARTMENT_MAP.get(issue_type_key)
    
    if not departments:
        is_mapped = False
        logger.info(f"⚠️ Unmapped issue type: '{issue_type}' (key: {issue_type_key})")
        
        # TIER 2: Check category
        category = ISSUE_CATEGORY_MAP.get(issue_type_key)
        
        if not category:
            # Fallback for completely unknown
            category = "public"
            logger.info("   Assuming 'public' category for unknown issue.")

        if category == "public":
            departments = ["general"]
        elif category == "private":
            departments = ["property_management"]
        else:
            departments = ["general"]
            
        # Create admin review entry logic
        mapping_review_entry = {
            "id": str(uuid.uuid4()),
            "case_id": ai_json.get("case_id"),
            "issue_type": issue_type,
            "submitted_description": ai_json.get("summary_explanation", ai_json.get("description", "")),
            "ai_confidence": ai_json.get("confidence", 0),
            "attempted_mapping": None,
            "current_routed_to": departments[0],
            "flagged_at": datetime.utcnow().isoformat(),
            "resolved": False,
            "admin_notes": f"Unmapped issue type '{issue_type}'. Routed to {departments[0]}. Needs categorization."
        }
        
        # Save to database
        await save_mapping_review(mapping_review_entry)
        
    # TIER 3: Get department contacts from zip code file
    zip_code_str = str(zip_code) if zip_code else ""
    authorities_data = ZIP_CODE_AUTHORITIES.get(zip_code_str)
    
    if not authorities_data:
        # Use default if zip not found
        logger.warning(f"⚠️ Zip code '{zip_code_str}' not found, using default.")
        authorities_data = ZIP_CODE_AUTHORITIES.get("default", {})
    
    # Get contacts for each department
    if departments:
        for dept in departments:
            dept_contacts = authorities_data.get(dept, [])
            if dept_contacts:
                authorities.extend(dept_contacts)
    
    # Fallback if no specific authorities found even after resolution
    if not authorities:
         logger.warning(f"⚠️ No authorities found for departments {departments} in zip {zip_code_str}. Trying general.")
         general = authorities_data.get("general", [])
         if general:
             authorities.extend(general)

    return {
        "authorities": authorities,
        "is_mapped": is_mapped,
        "mapping_review": mapping_review_entry,
        "departments": departments
    }

async def save_mapping_review(entry: Dict[str, Any]):
    try:
        db = await get_db()
        await db.authority_mapping_review.insert_one(entry)
        logger.info(f"📝 Saved unmapped issue review entry: {entry['issue_type']}")
    except Exception as e:
        logger.error(f"❌ Failed to save mapping review: {e}")

async def update_department_mapping(issue_type: str, departments: List[str], admin_email: str = "system"):
    """Update department mapping in memory and file."""
    issue_type_key = issue_type.lower().strip().replace(' ', '_')
    old_value = ISSUE_DEPARTMENT_MAP.get(issue_type_key)
    ISSUE_DEPARTMENT_MAP[issue_type_key] = departments
    
    # Save to file
    try:
        base_path = Path(__file__).parent.parent / "data"
        with open(base_path / 'issue_department_map.json', 'w') as f:
            json.dump(ISSUE_DEPARTMENT_MAP, f, indent=2)
            
        # Log Audit
        await log_audit_event(
            action="update_mapping",
            target=issue_type_key,
            details={"old": old_value, "new": departments},
            admin_email=admin_email
        )
        
        logger.info(f"✅ Updated mapping for {issue_type_key} -> {departments}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save issue_department_map.json: {e}")
        return False

def get_all_authorities():
    """Return the entire zip code authorities map."""
    return ZIP_CODE_AUTHORITIES

async def update_zip_authority(zip_code: str, data: Dict[str, Any], admin_email: str = "system"):
    """Update or add a zip code entry."""
    zip_str = str(zip_code)
    old_value = ZIP_CODE_AUTHORITIES.get(zip_str)
    
    ZIP_CODE_AUTHORITIES[zip_str] = data
    success = _save_zip_authorities()
    
    if success:
        await log_audit_event(
            action="update_zip_authority",
            target=zip_str,
            details={"old": old_value, "new": data},
            admin_email=admin_email
        )
        
    return success

def _save_zip_authorities():
    try:
        base_path = Path(__file__).parent.parent / "data"
        with open(base_path / 'zip_code_authorities.json', 'w') as f:
            json.dump(ZIP_CODE_AUTHORITIES, f, indent=2)
        logger.info(f"✅ Saved zip_code_authorities.json")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save zip_code_authorities.json: {e}")
        return False

def get_all_department_mappings():
    """Return the entire issue -> department map."""
    return ISSUE_DEPARTMENT_MAP

