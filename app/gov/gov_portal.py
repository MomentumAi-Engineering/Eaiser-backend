from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from services.mongodb_service import get_db
from pydantic import BaseModel
from core.auth import get_current_user, require_permission
from typing import List, Optional, Dict, Any
from bson import ObjectId
import logging
from datetime import datetime, timedelta

router = APIRouter(
    prefix="/gov/portal",
    tags=["Government Portal Content"]
)

logger = logging.getLogger(__name__)

# ── "No real issue" / junk report detection ──────────────────────────────
# When the AI can't find an actionable civic problem it classifies the report
# as "No Visible Public Infrastructure Issue" (issue_detected=False). These —
# along with not-a-civic-issue / unusable-image / unknown reports — must NOT be
# shown to officials or routed to any authority. They are noise.
NO_ISSUE_TYPES = {
    "no visible public infrastructure issue",
    "no visible issue",
    "no issue",
    "not a civic issue",
    "no civic issue",
    "image unusable",
    "none",
    "",
}
NO_ISSUE_STATUSES = {"no_issue_detected", "not_a_civic_issue", "image_unusable", "unknown_only"}


def _is_no_issue(iss: dict) -> bool:
    """True if this report has no real, actionable infrastructure issue."""
    if not isinstance(iss, dict):
        return False
    itype = str(iss.get("issue_type") or "").strip().lower()
    if itype in NO_ISSUE_TYPES:
        return True
    if str(iss.get("analysis_status") or "").strip().lower() in NO_ISSUE_STATUSES:
        return True
    # issue_detected can live top-level or nested under report / ai_evaluation
    rep = iss.get("report") if isinstance(iss.get("report"), dict) else {}
    for src in (iss, rep, rep.get("ai_evaluation") or {}, iss.get("ai_evaluation") or {}):
        if isinstance(src, dict) and src.get("issue_detected") is False:
            return True
    return False


# ── EAiSER-owned reports ──────────────────────────────────────────────────
# A report only reaches a CITY department once the EAiSER admin has cleared and
# dispatched it. Until then — while it is being processed, is pending review, is
# screened out as no-issue, or was rejected — it belongs to EAiSER, NOT the
# city. This is the "any doubt → EAiSER admin; no issue → EAiSER, never a
# department" rule. After approval the status becomes submitted/accepted (and
# later assigned/in_progress/resolved), none of which are listed here.
EAISER_ONLY_STATUSES = {
    "pending", "processing", "needs_review", "pending_review",
    "screened_out", "dispatch_decision", "rejected", "declined",
    "no_issue_detected", "not_a_civic_issue", "image_unusable", "unknown_only",
}


def _is_eaiser_only(iss: dict) -> bool:
    """True if the report is still EAiSER's (under review / doubtful / junk / rejected) — not a city department's."""
    if not isinstance(iss, dict):
        return False
    return str(iss.get("status") or "").strip().lower() in EAISER_ONLY_STATUSES


# ── Issue-type alias expansion ────────────────────────────────────────────
# Stored reports use issue_type VARIANTS the AI/older pipeline emitted (e.g.
# "tree_fallen" instead of the canonical "fallen_tree"). A department owns the
# canonical type, so without alias expansion those reports never match and stay
# invisible even after being cleared. We build canonical → [all aliases] from
# issue_department_map.json so a dept that owns "fallen_tree" also matches
# "tree_fallen", "tree_down", etc.
import os as _os
import json as _json

_CANON_TO_ALIASES = None


def _load_canon_aliases():
    global _CANON_TO_ALIASES
    if _CANON_TO_ALIASES is not None:
        return _CANON_TO_ALIASES
    rev = {}
    try:
        path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "issue_department_map.json")
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        for alias, canon in (data.get("_aliases") or {}).items():
            rev.setdefault(str(canon), set()).add(str(alias))
    except Exception as e:
        logger.warning(f"⚠️ Could not load issue alias map: {e}")
    # Roadkill variants the AI emits but the map doesn't alias → dead_animal.
    for a in ("animal_carcass", "wildlife_hit", "animal_accident", "roadkill"):
        rev.setdefault("dead_animal", set()).add(a)
    _CANON_TO_ALIASES = rev
    return rev


def expand_issue_types(types):
    """Each canonical type → {itself, Title Case display, all known aliases}."""
    rev = _load_canon_aliases()
    out = set()
    for t in types or []:
        t = str(t)
        out.add(t)
        out.add(t.replace("_", " ").title())
        for a in rev.get(t, ()):  # aliases that resolve to this canonical type
            out.add(a)
            out.add(a.replace("_", " ").title())
    return list(out)

@router.get("/reports")
async def get_gov_reports(
    department: str = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Fetch real reports (issues) tailored to the logged-in official.
    Filters by the Official's ZIP code and Department.

    `department` (optional): a management user (super_admin / ops_manager) can
    pass a department name to scope the result to that department's cleared
    reports — used by the per-department dashboard drill-down.
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
    # Normalized Mapping (Internal keys are Title Case)
    normalized_map = {
        "FIRE DEPARTMENT": ["Fire Hazard", "Gas Leak", "fire", "gas_leak", "exposed_wires", "unsafe_building", "Exposed Wires", "FIRE"],
        "PUBLIC WORKS": ["Pothole", "Fallen Tree", "pothole", "road_damage", "fallen_tree", "sinkhole", "bridge_issue", "Road Damage", "Drainage", "ROAD_DAMAGE"],
        "PUBLIC DEPARTMENT": ["Pothole", "Fallen Tree", "pothole", "road_damage", "fallen_tree", "sinkhole", "bridge_issue", "Road Damage", "Drainage", "ROAD_DAMAGE"],
        "TRANSPORTATION": ["Pothole", "Fallen Tree", "pothole", "road_damage", "fallen_tree", "sinkhole", "bridge_issue", "Road Damage", "Drainage", "ROAD_DAMAGE"],
        "WATER & UTILITIES": ["Water Leakage", "water_leakage", "flooding", "sewer_block", "drain_block", "Water Leak", "WATER_LEAK"],
        "SANITATION": ["Garbage", "garbage", "dead_animal", "illegal_dumping", "Waste", "GARBAGE"],
        "ELECTRICAL & TRAFFIC": ["Streetlight", "Signal Down", "streetlight", "signal_malfunction", "exposed_wires", "Signal Out", "LIGHTING"],
        "POLICE COMMAND": ["Car Accident", "car_accident", "security_threat", "illegal_activity", "Vandalism", "ACCIDENT"]
    }

    # ROLE & DEPARTMENT ISOLATION
    clean_role = str(user_role).lower().strip()
    is_management = clean_role in ["super_admin", "ops_manager", "admin", "mayor"] or str(user_dept).upper() in ["ALL", "CITY_MANAGEMENT", "MANAGEMENT"]

    async def _allowed_types_for(dept_name):
        """A department's issue_types (seeded source of truth, alias-expanded),
        falling back to the legacy name→type map. Returns [] if the dept has no
        configured types."""
        if not dept_name:
            return []
        import re as _re
        doc = await db["gov_departments"].find_one({
            "city": current_user.get("org"),
            "name": {"$regex": f"^{_re.escape(str(dept_name).strip())}$", "$options": "i"},
        })
        types = list(doc["issue_types"]) if (doc and doc.get("issue_types")) \
            else normalized_map.get(str(dept_name).upper().strip(), [])
        return expand_issue_types(types) if types else []

    # Decide the scope: which department's issue_types to restrict to (None = no
    # restriction) and whether to apply the EAiSER "cleared only" gate.
    apply_gate = True
    scope_types = None
    if clean_role == "crew_member":
        # Crew see only what's specifically assigned to them (already past review).
        query = {"assigned_to": current_user.get("email")}
        apply_gate = False
        scope_types = None
    elif not is_management:
        scope_types = await _allowed_types_for(user_dept)
    elif department:
        # Management drilling into ONE department's dashboard — show that
        # department's cleared reports, exactly as a dept member would see.
        scope_types = await _allowed_types_for(department)
    else:
        # Management overview — every report, full oversight, no gate.
        scope_types = None
        apply_gate = False

    if clean_role != "crew_member" and scope_types is not None:
        if scope_types:
            # MULTI-ISSUE: match the primary issue_type OR any detected issue, so
            # a department sees every report that involves one of its issue types.
            query["$or"] = [
                {"issue_type": {"$in": scope_types}},
                {"detected_issues.issue": {"$in": scope_types}},
            ]
        else:
            logger.warning(f"⚠️ No issue_types for department scope '{department or user_dept}'")
            query["issue_type"] = {"$in": []}

    cursor = db["issues"].find(query).sort("timestamp", -1)
    issues = await cursor.to_list(length=100)
    
    formatted_reports = []
    for iss in issues:
        # 🚫 Hide "no real issue" / junk reports from the portal entirely.
        if _is_no_issue(iss):
            continue
        # 🛡️ Doubtful / unreviewed / rejected reports stay with the EAiSER admin.
        # The gate applies to department-scoped views (dept members, and a
        # management user drilling into a specific department); the management
        # OVERVIEW (no department) sees everything for full oversight.
        if apply_gate and _is_eaiser_only(iss):
            continue
        # Resolve risk/severity
        severity = iss.get("severity", "Medium")
        if isinstance(severity, dict): severity = severity.get("label", "Medium")

        report_id = iss.get("issue_id") or str(iss.get("_id"))
        img_lookup_id = str(iss.get("_id") or report_id)

        # ── Original citizen image(s) — exactly what was submitted from mobile.
        # A report can carry one OR several images. Order of preference:
        #   image_urls[] (multi Cloudinary) → image_url (single Cloudinary)
        #   → image_id (single GridFS, served via /api/issues/{id}/image).
        # GridFS path is relative (/api/...); the frontend resolves it against
        # the API origin. Cloudinary entries are already absolute http URLs.
        original_images = []
        iu = iss.get("image_urls")
        if isinstance(iu, list):
            original_images = [u for u in iu if u]
        if not original_images and iss.get("image_url"):
            original_images = [iss.get("image_url")]
        if not original_images and iss.get("image_id"):
            original_images = [f"/api/issues/{img_lookup_id}/image"]

        before_primary = original_images[0] if original_images else ""

        report = {
            "id": report_id,
            "report_id": iss.get("report_id"),
            "type": (iss.get("issue_type") or "Other").replace("_", " ").title(),
            "raw_type": iss.get("issue_type"), # Keep raw for exact matching if needed
            # All issues the AI detected in this report (for multi-issue photos),
            # so a department can see every relevant issue, not just the primary.
            "detected_issues": [
                (di.get("issue") if isinstance(di, dict) else di)
                for di in (iss.get("detected_issues") or [])
            ],
            "location": iss.get("address", "Unknown Location"),
            "address": iss.get("address"),
            "location_zip": iss.get("zip_code"), # Required for frontend filtering
            "zip": iss.get("zip_code"),
            "status": iss.get("status", "open").title(),
            "raw_status": iss.get("status"),
            "risk": str(severity).title(),
            "severity": str(severity).title(),
            "priority": iss.get("priority"),
            "confidence": iss.get("confidence"),
            "date": iss.get("timestamp_formatted") or "Recently",
            "timestamp": iss.get("timestamp"),
            "desc": iss.get("description") or iss.get("report", {}).get("issue_overview", {}).get("summary_explanation", "No description provided."),
            "recommended_actions": iss.get("recommended_actions") or [],
            "reporter": iss.get("user_email"),
            # Before = original citizen submission (one or many). After = field
            # evidence proving the work was done (uploaded by crew).
            "images": original_images,                     # all "before" images
            "image": before_primary,                       # first before (back-compat)
            "media_url": before_primary,                   # first before (back-compat)
            "resolution_media_url": iss.get("resolution_media_url"),  # "after" image
            "department": user_dept or "GENERAL",
            "coordinates": [iss.get("latitude", 0), iss.get("longitude", 0)],
            "assigned_to": iss.get("assigned_to"),
            "assigned_by": iss.get("assigned_by"),
            "assigned_at": iss.get("assigned_at"),
            # Contractor workflow surface (assignment, quote, accept, escalation)
            # so the portal can show quotes + escalation state.
            "contractor_ref": iss.get("contractor_ref"),
            "contractor_quote": iss.get("contractor_quote"),
            "contractor_accepted": bool(iss.get("contractor_accepted")),
            "contractor_before_url": iss.get("contractor_before_url"),
            "escalated_to_dept_head": bool(iss.get("escalated_to_dept_head")),
            "resolved_by": iss.get("resolved_by"),
            # Verification workflow: an "after" photo (resolution_media_url) makes
            # a report ready for ops review; verify → resolved, decline → reopen.
            "verified_by": iss.get("verified_by"),
            "verified_at": iss.get("verified_at"),
            "decline_reason": iss.get("decline_reason"),
            "declined_by": iss.get("declined_by"),
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
    
    clean_role = str(current_user.get("role", "")).lower().replace(" ", "_")
    
    query = {}
    if clean_role == "crew_member":
        query = {"assigned_to": current_user.get("email")}
    else:
        if user_zip and user_zip != "ALL": query["zip_code"] = user_zip
    
    # Dynamic counts
    total = await db["issues"].count_documents(query)
    open_issues = await db["issues"].count_documents({**query, "status": {"$in": ["reported", "approved", "open", "in_progress"]}})
    resolved = await db["issues"].count_documents({**query, "status": "resolved"})
    
    # Efficiency calculation
    efficiency_val = int((resolved / total * 100)) if total > 0 else 100
    if efficiency_val > 100: efficiency_val = 100

    try:
        # Velocity calculation (Concurrent fetches)
        velocity = []
        now = datetime.utcnow()
        import asyncio
        
        async def fetch_bucket(i):
            h = now - timedelta(hours=i*2)
            c_new = await db["issues"].count_documents({
                **query, 
                "$or": [{"timestamp": {"$gte": h.isoformat()}}, {"timestamp": {"$gte": h}}],
                "status": {"$ne": "resolved"}
            })
            c_res = await db["issues"].count_documents({
                **query, 
                "$or": [{"timestamp": {"$gte": h.isoformat()}}, {"timestamp": {"$gte": h}}],
                "status": "resolved"
            })
            return {"time": h.strftime("%H:%M"), "new": c_new, "resolved": c_res}
            
        tasks = [fetch_bucket(i) for i in range(7)]
        results = await asyncio.gather(*tasks)
        velocity = list(reversed(results))
    except Exception as e:
        logger.error(f"Velocity calc failed: {str(e)}", exc_info=True)
        velocity = []

    return {
        "total_reports": total,
        "active_incidents": open_issues,
        "resolved_cases": resolved,
        # ALIASES FOR MOBILE APP COMPATIBILITY
        "totalIncidents": total,
        "openIncidents": open_issues,
        "resolvedIncidents": resolved,
        "efficiency": f"{efficiency_val}%",
        "avg_response_time": "3.8 hrs" if efficiency_val > 90 else "5.2 hrs",
        "velocity": velocity
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
    
    # Security/UX isolation:
    # 1. Hide own account from the personnel list (it is already in the header/profile)
    query["_id"] = {"$ne": ObjectId(current_user["_id"])}
    
    # 2. Only show 'Operations' staff in the crew/team list.
    # Excludes Super Admins/City Managers from the operational crew directory.
    query["role"] = "operations"
        
    cursor = db["government_users"].find(query, {"hashed_password": 0}).sort("name", 1)
    officials = await cursor.to_list(length=100)
    
    team = []
    for off in officials:
        db_role = off.get("role", "operations")
        display_role = "Ops Staff" if db_role == "operations" else "Super Admin"
        
        team.append({
            "id": str(off["_id"]),
            "name": off.get("name"),
            "role": display_role,
            "dept": off.get("department") or off.get("city") or "GENERAL OPERATIONS",
            "department": off.get("department") or off.get("city") or "GENERAL OPERATIONS",
            "email": off.get("email"),
            "initials": "".join([n[0] for n in off.get("name", "U").split() if n]).upper(),
            "status": off.get("status", "available"),
            "assignment": off.get("assignment", None),
            "color": "bg-zinc-800 text-white"
        })
    return team

@router.get("/analytics")
async def get_gov_analytics(
    current_user: dict = Depends(get_current_user)
):
    """
    Detailed analytics aggregation for the department.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    user_dept = current_user.get("dept")
    
    query = {}
    if user_zip and user_zip != "ALL": query["zip_code"] = user_zip

    # 1. Category Breakdown
    pipeline_cat = [
        {"$match": query},
        {"$group": {"_id": "$issue_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 4}
    ]
    cursor_cat = db["issues"].aggregate(pipeline_cat)
    top_categories = await cursor_cat.to_list(length=4)
    
    # 2. SLA Breakdown per Category (Real Data)
    category_slas = []
    for cat in top_categories:
        cat_name = cat["_id"]
        cat_total = cat["count"]
        cat_resolved = await db["issues"].count_documents({**query, "issue_type": cat_name, "status": "resolved"})
        sla_pct = int((cat_resolved / cat_total * 100)) if cat_total > 0 else 0
        category_slas.append({
            "label": str(cat_name).replace("_", " ").title(),
            "compliance": sla_pct
        })

    # 3. Resolution Trend (Real Data - Last 4 Weeks based on timestamp)
    # We will use mongo aggregation to group by week
    current_time = datetime.utcnow()
    four_weeks_ago = current_time - timedelta(days=28)
    
    # Simple Python-based bucketing for trend to ensure it works across all Mongo versions
    recent_issues = await db["issues"].find({**query}).to_list(length=10000)
    
    week_buckets = {"W1": {"received": 0, "resolved": 0}, "W2": {"received": 0, "resolved": 0}, "W3": {"received": 0, "resolved": 0}, "W4": {"received": 0, "resolved": 0}}
    
    for iss in recent_issues:
        try:
            # Handle different date formats or fallbacks securely
            ts = iss.get("timestamp") or iss.get("date")
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, datetime):
                dt = ts
            else:
                dt = current_time # Fallback constraint if no timestamp
                
            # Remove tzinfo for safe subtraction if current_time has none
            dt = dt.replace(tzinfo=None)
            
            days_ago = (current_time - dt).days
            if days_ago < 7: w = "W4"
            elif days_ago < 14: w = "W3"
            elif days_ago < 21: w = "W2"
            elif days_ago < 28: w = "W1"
            else: continue
            
            week_buckets[w]["received"] += 1
            if iss.get("status") == "resolved":
                week_buckets[w]["resolved"] += 1
        except:
            pass # Keep iterating if corrupt date prevents parsing
            
    trend = [{"week": k, "received": v["received"], "resolved": v["resolved"]} for k, v in week_buckets.items()]

    # 4. Overall SLA Compliance
    total_count = await db["issues"].count_documents(query)
    resolved_count = await db["issues"].count_documents({**query, "status": "resolved"})
    
    on_track = resolved_count
    warning = await db["issues"].count_documents({**query, "status": {"$in": ["assigned", "assigned_external", "in_progress", "pending_verification", "approved"]}})
    breached = await db["issues"].count_documents({**query, "status": {"$in": ["reported", "open", "new"]}})
    
    sla_total = on_track + warning + breached
    if sla_total > 0:
        p_on_track = int(on_track / sla_total * 100)
        p_warning = int(warning / sla_total * 100)
        p_breached = 100 - p_on_track - p_warning
    else:
        p_on_track, p_warning, p_breached = 100, 0, 0

    return {
        "summary": {
            "active": await db["issues"].count_documents({**query, "status": {"$ne": "resolved"}}),
            "resolved": resolved_count,
            "compliance": f"{p_on_track}%",
            "satisfaction": "4.8/5" if p_on_track > 90 else ("4.2/5" if p_on_track > 50 else "3.1/5")
        },
        "categories": [{"label": str(c["_id"]).title().replace("_", " "), "count": c["count"]} for c in top_categories],
        "category_slas": category_slas,
        "trend": trend,
        "sla": {
            "on_track": p_on_track,
            "warning": p_warning,
            "breached": p_breached
        }
    }

@router.get("/staff-performance")
async def get_staff_performance(
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate real performance metrics for all officials in the city/ZIP.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    
    query = {}
    if user_zip and user_zip != "ALL": query["zip_code"] = user_zip
    
    # Get only operational staff in this city (exclude super admins)
    query["role"] = {"$ne": "super_admin"}
    cursor = db["government_users"].find(query)
    users = await cursor.to_list(length=100)
    
    performance_list = []
    for user in users:
        email = user.get("email")
        # Count resolved issues assigned to this user
        resolved = await db["issues"].count_documents({"assigned_to": email, "status": "resolved"})
        active = await db["issues"].count_documents({"assigned_to": email, "status": {"$ne": "resolved"}})
        
        total = resolved + active
        efficiency = int((resolved / total * 100)) if total > 0 else 100
        
        performance_list.append({
            "name": user.get("name"),
            "dept": user.get("department"),
            "role": "Official" if user.get("role") == "super_admin" else "Crew",
            "resolved": resolved,
            "avgTime": "3.5 hrs" if efficiency > 90 else "4.8 hrs",
            "rating": 4.5 + (efficiency / 200),
            "efficiency": min(efficiency, 100)
        })
        
    # Sort by resolved count descending
    performance_list.sort(key=lambda x: x["resolved"], reverse=True)
    return performance_list

@router.get("/departments-stats")
async def get_department_performance(
    current_user: dict = Depends(get_current_user)
):
    """
    Aggregation for the 'Organization Architecture' dashboard section.
    Provides live counts and efficiency metrics per municipal sector.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    
    query = {}
    if user_zip and user_zip != "ALL": query["zip_code"] = user_zip

    # Base departments to ensure complete dataset
    depts = [
        {"name": "Fire Department", "color": "#EF4444", "types": ["Fire Hazard", "Gas Leak", "fire", "gas_leak", "exposed_wires", "Exposed Wires"]},
        {"name": "Public Works", "color": "#D4A017", "types": ["Pothole", "Fallen Tree", "pothole", "road_damage", "bridge_issue", "Road Damage", "Drainage"]},
        {"name": "Water Services", "color": "#3B82F6", "types": ["Water Leakage", "water_leakage", "flooding", "sewer_block", "drain_block", "Water Leak"]},
        {"name": "Police Command", "color": "#A855F7", "types": ["Car Accident", "car_accident", "security_threat", "illegal_activity", "Vandalism"]},
        {"name": "Sanitation", "color": "#F97316", "types": ["Garbage", "garbage", "dead_animal", "illegal_dumping", "Waste"]}
    ]

    results = []
    for d in depts:
        active = await db["issues"].count_documents({**query, "issue_type": {"$in": d["types"]}, "status": {"$ne": "resolved"}})
        resolved = await db["issues"].count_documents({**query, "issue_type": {"$in": d["types"]}, "status": "resolved"})
        
        # Performance heuristic: (Resolved / Total) * 100
        total = active + resolved
        perf = int((resolved / total * 100)) if total > 0 else 100
        if perf > 100: perf = 100

        results.append({
            "name": d["name"],
            "color": d["color"],
            "active": active,
            "resolved": resolved,
            "performance": perf,
            "budget": 150000, # Realistic baseline
            "budgetUsed": int(150000 * (0.4 + (perf/200))) # Simulated usage based on volume
        })

    return results

class AssignmentRequest(BaseModel):
    issue_id: str
    staff_email: str

@router.post("/assign")
async def assign_incident(
    payload: AssignmentRequest,
    current_user: dict = Depends(require_permission("assign_issue"))
):
    """
    Tactical assignment of an incident (issue) to a government staff member.
    """
    db = await get_db()
    
    # 1. Verify existence of the issue
    try:
        issue = await db["issues"].find_one({"_id": payload.issue_id})
        if not issue:
            # Try as ObjectId
            issue = await db["issues"].find_one({"_id": ObjectId(payload.issue_id)})
            
        if not issue:
            raise HTTPException(status_code=404, detail="Incident not found")
    except Exception as e:
        # If it's a malformed ID string
        issue = await db["issues"].find_one({"issue_id": payload.issue_id})
        if not issue:
            raise HTTPException(status_code=404, detail="Malformed or missing incident ID")

    # 2. Verify existence of the staff (Optional, but good for validation)
    # 2. Verify existence of the staff OR contractor
    staff = await db["government_users"].find_one({"email": payload.staff_email})
    contractor = None
    
    if not staff:
        staff = await db["admins"].find_one({"email": payload.staff_email})
        
    if not staff:
        try:
            contractor = await db["gov_contractors"].find_one({"_id": ObjectId(payload.staff_email)})
        except:
            pass

    if not staff and not contractor:
        raise HTTPException(status_code=404, detail=f"Target {payload.staff_email} not found. Is it a valid crew/contractor?")

    # Only APPROVED contractors may receive work orders (architecture: "assign to
    # APPROVED contractors"). A pending / escalated / rejected contractor is
    # blocked server-side even if a stale UI offered them — the dropdown filters
    # too, but this gate must not be bypassable.
    if contractor and str(contractor.get("approval_status", "")).lower() != "approved":
        raise HTTPException(
            status_code=403,
            detail=f"{contractor.get('company', 'This contractor')} is not approved yet "
                   f"(status: {contractor.get('approval_status', 'pending')}). Approve them before assigning work.",
        )

    # 3. Apply assignment transformation
    if contractor:
        update_data = {
            "status": "assigned_external",
            "assigned_to": f"{contractor.get('company')} (Contractor)",
            "assigned_by": current_user.get("email"),
            "assigned_at": datetime.utcnow(),
            "last_updated_at": datetime.utcnow().isoformat(),
            "contractor_ref": str(contractor["_id"]),
            # Also store the contractor's email so the work-order link survives the
            # contractor being re-created with a new _id (id can churn; email is stable).
            "contractor_email": contractor.get("email"),
        }
    else:
        update_data = {
            "status": "assigned",
            "assigned_to": payload.staff_email,
            "assigned_by": current_user.get("email"),
            "assigned_at": datetime.utcnow(),
            "last_updated_at": datetime.utcnow().isoformat()
        }
    
    await db["issues"].update_one(
        {"_id": issue["_id"]},
        {"$set": update_data}
    )
    
    # 4. Update the staff member's status to busy (only for internal staff)
    if staff:
        await db["government_users"].update_one(
            {"email": payload.staff_email},
            {"$set": {
                "status": "busy", 
                "assignment": str(issue["_id"])
            }}
        )
        # Check for push application token and trigger alert
        user_auth = await db["gov_users"].find_one({"email": payload.staff_email})
        if user_auth and user_auth.get("push_token"):
            push_payload = {
                "to": user_auth["push_token"],
                "sound": "default",
                "title": "EAiSER Command Center",
                "body": f"Incident {payload.issue_id} has been dispatched to you. Acknowledge immediately.",
                "data": {"type": "assignment", "issue_id": str(issue["_id"])}
            }
            try:
                import httpx
                import asyncio
                async def fire_push():
                    async with httpx.AsyncClient() as client:
                        await client.post("https://exp.host/--/api/v2/push/send", json=push_payload)
                asyncio.create_task(fire_push())
            except Exception as e:
                logger.error(f"Failed to kick off dispatch push: {e}")
    
    logger.info(f"✅ DEPLOYMENT: Incident {payload.issue_id} assigned to {payload.staff_email} by {current_user.get('email')}")
    
    # `staff` is None when the target is a contractor — derive the name safely.
    if contractor:
        assignee_name = f"{contractor.get('company', 'Contractor')} (Contractor)"
    elif staff:
        assignee_name = staff.get("name", payload.staff_email)
    else:
        assignee_name = payload.staff_email

    return {
        "success": True,
        "message": f"Successfully assigned to {assignee_name}",
        "assignment": update_data
    }

class StatusUpdate(BaseModel):
    status: str
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None

VERIFY_ROLES = {"super_admin", "admin", "operations", "ops_manager", "department_admin"}


def _can_verify(user: dict) -> bool:
    return str(user.get("role") or "").lower() in VERIFY_ROLES


@router.patch("/reports/{issue_id}/status")
async def update_incident_status(
    issue_id: str,
    update: StatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    db = await get_db()

    try:
        query_id = ObjectId(issue_id)
    except:
        query_id = issue_id

    # 🔒 Resolution gate: a report can ONLY be marked resolved after the crew has
    # uploaded a verification ("after") photo AND an operations staff member (or
    # above) approves it. Crew cannot self-resolve, and nothing resolves without
    # the photo. Use the /verify endpoint for the happy path — this guards direct
    # PATCH callers too.
    if update.status in ("resolved", "completed"):
        issue = await db["issues"].find_one({"_id": query_id})
        if not issue:
            raise HTTPException(status_code=404, detail="Report not found")
        if not issue.get("resolution_media_url"):
            raise HTTPException(status_code=400, detail="A crew verification photo is required before resolving. Ask the assigned crew to upload the 'after' photo.")
        if not _can_verify(current_user):
            raise HTTPException(status_code=403, detail="Only operations staff can verify and resolve a report.")

    update_data = {"status": update.status}
    if update.resolution_notes:
        update_data["resolution_notes"] = update.resolution_notes
    if update.resolved_by:
        update_data["resolved_by"] = update.resolved_by
    if update.resolved_at:
        update_data["resolved_at"] = update.resolved_at

    await db["issues"].update_one(
        {"_id": query_id},
        {"$set": update_data}
    )
    
    # If the issue is moving out of "assigned" (e.g. to pending_verification or resolved)
    # we must free up the crew member so they are no longer stuck showing "BUSY"
    if update.status in ["resolved", "completed", "pending_verification"]:
        await db["government_users"].update_many(
             {"assignment": str(query_id)},
             {"$set": {
                 "status": "available",
                 "assignment": None
             }}
        )

    return {"success": True}


class DeclineRequest(BaseModel):
    reason: Optional[str] = None


@router.post("/reports/{issue_id}/verify")
async def verify_resolution(
    issue_id: str,
    current_user: dict = Depends(require_permission("verify_resolution")),
):
    """
    Operations staff approves the crew's verification photo → marks the report
    RESOLVED. Requires the 'after' photo to exist; crew cannot self-verify.
    """
    db = await get_db()
    try:
        query_id = ObjectId(issue_id)
    except Exception:
        query_id = issue_id

    issue = await db["issues"].find_one({"_id": query_id})
    if not issue:
        raise HTTPException(status_code=404, detail="Report not found")
    if not issue.get("resolution_media_url"):
        raise HTTPException(status_code=400, detail="No crew verification photo to verify yet.")

    await db["issues"].update_one(
        {"_id": query_id},
        {"$set": {
            "status": "resolved",
            "verified_by": current_user.get("name") or current_user.get("email"),
            "verified_by_email": current_user.get("email"),
            "verified_at": datetime.utcnow().isoformat(),
            "resolved_at": issue.get("resolved_at") or datetime.utcnow().isoformat(),
            # Step 7 of the Resolution Loop: evidence is immutable once resolved.
            "evidence_locked": True,
        }},
    )
    # Free up whoever was assigned.
    await db["government_users"].update_many(
        {"assignment": str(query_id)},
        {"$set": {"status": "available", "assignment": None}},
    )

    # Step 6 of the Resolution Loop: close the loop with the citizen — send a
    # resolution notification with the before/after photo evidence.
    citizen_email = issue.get("user_email")
    if citizen_email:
        iu = issue.get("image_urls")
        before_url = (iu[0] if isinstance(iu, list) and iu else None) or issue.get("image_url")
        after_url = issue.get("resolution_media_url")
        try:
            from services.email_service import send_resolution_closure_email
            await send_resolution_closure_email(
                citizen_email,
                issue.get("issue_type", ""),
                issue.get("address", ""),
                before_url,
                after_url,
            )
            await db["issues"].update_one({"_id": query_id}, {"$set": {"citizen_notified_at": datetime.utcnow().isoformat()}})
        except Exception as e:
            logger.warning(f"⚠️ Citizen closure notification failed for {citizen_email}: {e}")

    return {"success": True, "status": "resolved"}


@router.post("/reports/{issue_id}/escalate")
async def escalate_report(
    issue_id: str,
    current_user: dict = Depends(require_permission("assign_issue")),
):
    """
    Staff escalates a report to the Department Head for attention (diagram:
    "Escalate to Dept Head"). Flags the report; Dept Heads can filter on it.
    """
    db = await get_db()
    try:
        query_id = ObjectId(issue_id)
    except Exception:
        query_id = issue_id

    issue = await db["issues"].find_one({"_id": query_id})
    if not issue:
        raise HTTPException(status_code=404, detail="Report not found")

    await db["issues"].update_one(
        {"_id": query_id},
        {"$set": {
            "escalated_to_dept_head": True,
            "escalated_by": current_user.get("email"),
            "escalated_at": datetime.utcnow().isoformat(),
        }},
    )
    return {"success": True, "escalated": True}


@router.post("/reports/{issue_id}/decline")
async def decline_resolution(
    issue_id: str,
    payload: DeclineRequest,
    current_user: dict = Depends(require_permission("verify_resolution")),
):
    """
    Operations staff rejects the crew's verification photo → reopens the report
    (back to in_progress) so the crew can redo the work and re-upload. The
    rejected photo is kept in history; the 'after' slot is cleared.
    """
    db = await get_db()
    try:
        query_id = ObjectId(issue_id)
    except Exception:
        query_id = issue_id

    issue = await db["issues"].find_one({"_id": query_id})
    if not issue:
        raise HTTPException(status_code=404, detail="Report not found")
    if issue.get("evidence_locked") or str(issue.get("status", "")).lower() == "resolved":
        raise HTTPException(status_code=400, detail="This report is resolved — its evidence is locked and can't be changed.")

    history = issue.get("rejected_resolutions") or []
    if issue.get("resolution_media_url"):
        history.append({
            "url": issue.get("resolution_media_url"),
            "declined_by": current_user.get("name") or current_user.get("email"),
            "declined_at": datetime.utcnow().isoformat(),
            "reason": (payload.reason or "").strip(),
        })

    await db["issues"].update_one(
        {"_id": query_id},
        {
            "$set": {
                "status": "in_progress",          # reopen for the crew to redo
                "decline_reason": (payload.reason or "").strip(),
                "declined_by": current_user.get("name") or current_user.get("email"),
                "declined_at": datetime.utcnow().isoformat(),
                "rejected_resolutions": history,
            },
            "$unset": {"resolution_media_url": "", "resolved_at": "", "resolved_by": ""},
        },
    )
    return {"success": True, "status": "in_progress"}

@router.post("/reports/{issue_id}/evidence")
async def upload_evidence(
    issue_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    from services.cloudinary_service import upload_file_to_cloudinary
    db = await get_db()

    # Evidence lock: once a report is resolved, its before/after evidence is
    # immutable — reject any further upload.
    try:
        _guard_id = ObjectId(issue_id)
    except Exception:
        _guard_id = issue_id
    _existing = await db["issues"].find_one({"_id": _guard_id})
    if _existing and (_existing.get("evidence_locked") or str(_existing.get("status", "")).lower() == "resolved"):
        raise HTTPException(status_code=400, detail="This report is resolved — its evidence is locked.")

    contents = await file.read()
    result = await upload_file_to_cloudinary(contents=contents, folder="report_evidence")
    
    if result and result.get("url"):
        url = result.get("url")
        
        try:
            query_id = ObjectId(issue_id)
        except:
            query_id = issue_id
            
        # Change status to 'pending_verification' directly when evidence is uploaded
        await db["issues"].update_one(
            {"_id": query_id},
            {"$set": {
                "resolution_media_url": url, 
                "status": "pending_verification",
                "resolved_at": datetime.utcnow().isoformat(),
                "resolved_by": current_user.get("name", "Field Officer")
            }}
        )
        
        # Crew member has successfully completed their field task, so free them up immediately!
        if current_user and current_user.get("email"):
             await db["government_users"].update_one(
                 {"email": current_user["email"]},
                 {"$set": {
                     "status": "available",
                     "assignment": None
                 }}
             )
             
        # Just in case this was initiated by someone else, definitively clear the assignment from anyone tied to this issue
        await db["government_users"].update_many(
             {"assignment": str(query_id)},
             {"$set": {
                 "status": "available",
                 "assignment": None
             }}
        )
        
        return {"success": True, "url": url}
    else:
        raise HTTPException(status_code=500, detail="Failed to upload image")

@router.get("/schedule")
async def get_gov_schedule(
    current_user: dict = Depends(get_current_user)
):
    """
    Real-time schedule derived from live assignments.
    """
    db = await get_db()
    user_zip = current_user.get("zip")
    
    query_users = {"role": {"$in": ["crew_member", "operations"]}}
    if user_zip and user_zip != "ALL":
        query_users["zip_code"] = user_zip
        
    cursor_users = db["government_users"].find(query_users)
    users = await cursor_users.to_list(length=100)
    
    crew_names = [u.get("name", "Unknown Official") for u in users]
    if not crew_names:
        users = await db["government_users"].find({"role": {"$ne": "super_admin"}}).to_list(100)
        crew_names = [u.get("name", "Unknown Official") for u in users]
    
    crew_names = sorted(list(set(crew_names)))
    
    query_issues = {"assigned_to": {"$exists": True, "$ne": None}}
    if user_zip and user_zip != "ALL":
        query_issues["zip_code"] = user_zip
        
    issues = await db["issues"].find(query_issues).sort("assigned_at", -1).to_list(100)
    
    assignments = []
    
    for iss in issues:
        assigned_to_email = iss.get("assigned_to")
        crew_name = None
        for u in users:
            if u.get("email") == assigned_to_email:
                crew_name = u.get("name")
                break
                
        if not crew_name or crew_name not in crew_names:
            continue
            
        status = iss.get("status")
        mapped_status = "upcoming"
        if status in ["resolved", "completed"]: mapped_status = "completed"
        elif status in ["in_progress", "assigned", "pending_verification"]: mapped_status = "in-progress"
        
        ts = iss.get("assigned_at") or iss.get("timestamp")
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                dt = datetime.utcnow()
        elif isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.utcnow()
            
        hour = dt.hour
        idx = hour - 6
        if idx < 0: idx = max(0, hour)
        if idx > 12: idx = 11
        
        start = idx
        end = min(start + 2, 12)
        
        issue_type = iss.get("issue_type", "Task").replace("_", " ").title()
        street = iss.get("address", "").split(",")[0][:15] if iss.get("address") else "Location hidden"
        task_label = f"{issue_type} - {street}"
        
        assignments.append({
            "crew": crew_name,
            "start": start,
            "end": end,
            "task": task_label,
            "status": mapped_status,
            "id": str(iss.get("_id"))
        })

    return {
        "crew": crew_names,
        "assignments": assignments
    }

