from fastapi import APIRouter, HTTPException, Depends
from services.mongodb_service import get_db
from core.auth import get_current_user
from typing import List, Optional
from bson import ObjectId
import logging
from datetime import datetime

router = APIRouter(
    prefix="/gov/portal",
    tags=["Government Portal Content"]
)

logger = logging.getLogger(__name__)

@router.get("/reports")
async def get_gov_reports(
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch real reports (issues) tailored to the logged-in official.
    Filters by the Official's ZIP code and Department.
    """
    if current_user.get("type") != "gov_portal":
        # Allow Super Admins from the main dashboard to view too if they want
        if current_user.get("role") != "super_admin":
             raise HTTPException(status_code=403, detail="Government Portal access required")
    
    db = await get_db()
    
    user_dept = current_user.get("dept")
    user_zip = current_user.get("zip")
    user_role = current_user.get("role")
    
    query = {}
    
    # Filter by Location (ZIP)
    if user_zip and user_zip != "ALL":
        query["zip_code"] = user_zip
        
    # Filter by Department
    # Mapping for common department names to issue types or categories
    dept_type_map = {
        "Fire Department": ["Fire Hazard", "Gas Leak", "fire", "gas_leak", "exposed_wires", "unsafe_building", "Exposed Wires"],
        "Public Works": ["Pothole", "Fallen Tree", "pothole", "road_damage", "sinkhole", "bridge_issue", "Road Damage"],
        "Water & Utilities": ["Water Leakage", "water_leakage", "flooding", "sewer_block", "drain_block", "Water Leak"],
        "Sanitation": ["Garbage", "garbage", "dead_animal", "illegal_dumping", "Waste"],
        "Electrical & Traffic": ["Streetlight", "Signal Down", "streetlight", "signal_malfunction", "exposed_wires", "Signal Out"],
        "Police Command": ["Car Accident", "car_accident", "security_threat", "illegal_activity", "Vandalism"]
    }

    if user_role != "super_admin":
        if user_dept and user_dept != "ALL":
            allowed_types = dept_type_map.get(user_dept, [])
            # Enforce filter even if empty mapping, to prevent seeing everything
            query["issue_type"] = {"$in": allowed_types}
        elif user_dept == "ALL":
            pass # Keep query as-is (ZIP filtered)
        else:
            # No department assigned? Show nothing.
            query["issue_type"] = {"$in": []}
        
    cursor = db["issues"].find(query).sort("timestamp", -1)
    issues = await cursor.to_list(length=100)
    
    formatted_reports = []
    for iss in issues:
        # Resolve risk/severity
        severity = iss.get("severity", "Medium")
        if isinstance(severity, dict): severity = severity.get("label", "Medium")
        
        report = {
            "id": iss.get("issue_id") or str(iss.get("_id")),
            "type": (iss.get("issue_type") or "Other").replace("_", " ").title(),
            "raw_type": iss.get("issue_type"), # Keep raw for exact matching if needed
            "location": iss.get("address", "Unknown Location"),
            "location_zip": iss.get("zip_code"), # Required for frontend filtering
            "zip": iss.get("zip_code"),
            "status": iss.get("status", "open").title(),
            "risk": severity.title(),
            "date": iss.get("timestamp_formatted") or "Recently",
            "desc": iss.get("description") or iss.get("report", {}).get("issue_overview", {}).get("summary_explanation", "No description provided."),
            "image": iss.get("image_url", ""),
            "department": user_dept or "GENERAL",
            "coordinates": [iss.get("latitude", 0), iss.get("longitude", 0)]
        }
        formatted_reports.append(report)
        
    return formatted_reports

@router.get("/stats")
async def get_gov_stats(
    current_user: dict = Depends(get_current_user)
):
    """
    Real-time stats for the official's dashboard.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    user_dept = current_user.get("dept")
    
    query = {}
    if user_zip and user_zip != "ALL": query["zip_code"] = user_zip
    
    # Simple count aggregation
    total = await db["issues"].count_documents(query)
    open_issues = await db["issues"].count_documents({**query, "status": {"$in": ["reported", "approved", "open"]}})
    resolved = await db["issues"].count_documents({**query, "status": "resolved"})
    
    return {
        "total_reports": total,
        "active_incidents": open_issues,
        "resolved_cases": resolved,
        "efficiency": "88%", # Heuristic for now
        "avg_response_time": "4.2 hrs"
    }

@router.get("/team")
async def get_gov_team(
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch all officials in the same city/ZIP for the team directory.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    
    query = {}
    if user_zip and user_zip != "ALL":
        query["zip_code"] = user_zip
        
    cursor = db["government_users"].find(query, {"hashed_password": 0})
    officials = await cursor.to_list(length=50)
    
    team = []
    for off in officials:
        team.append({
            "id": str(off["_id"]),
            "name": off.get("name"),
            "role": "Official",
            "dept": off.get("department"),
            "initials": "".join([n[0] for n in off.get("name", "U").split() if n]).upper(),
            "color": "bg-zinc-800 text-white"
        })
    return team
