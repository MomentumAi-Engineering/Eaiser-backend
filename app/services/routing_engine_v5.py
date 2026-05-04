"""
EAiSER V5 Routing Engine
=========================
Complete 16-Phase routing pipeline implementing the V5 spec.

Phases:
  0. Definitions (P0-P3 severity, hard rules)
  1. Input validation
  2. ZIP Gate (service area check)
  3. AI Analysis (delegates to existing SIMI/MIMI)
  4. City State (LIVE/NOT_LIVE)
  5. Routing Engine (6A: Tier0 safety, 6B: dept mapping, 6C: dedup)
  6. Build recipient list
  7-8. User interaction (handled by iOS app)
  9. Hard guardrail (zero-recipient block)
  10. Persistence
  11. Dispatch
  12-16. Confirmation, ops, feedback, close
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# PHASE 0 — DEFINITIONS (NON-NEGOTIABLE)
# ═══════════════════════════════════════════════════════════════

# Severity mapping: P0-P3
SEVERITY_MAP = {
    "P0": {"label": "Critical", "description": "Life/Safety", "priority_ordinal": 4},
    "P1": {"label": "High", "description": "Urgent", "priority_ordinal": 3},
    "P2": {"label": "Medium", "description": "Moderate", "priority_ordinal": 2},
    "P3": {"label": "Low", "description": "Low", "priority_ordinal": 1},
}

# Issue → Severity (P-level) mapping
ISSUE_SEVERITY_MAP = {
    # Tier 0 → P0 (life/safety)
    "Fire Hazard": "P0",
    "Car Accident": "P0",
    "Downed Power Line": "P0",
    # Tier 1 → P1 (urgent)
    "Flooding / Standing Water": "P1",
    "Open / Damaged Manhole": "P1",
    "Broken Traffic Signal": "P1",
    "Fallen / Hazardous Tree": "P1",
    # Tier 2 → P2 (moderate)
    "Pothole": "P2",
    "Road Damage": "P2",
    "Broken Streetlight": "P2",
    "Water Leakage": "P2",
    "Damaged Sidewalk": "P2",
    "Damaged Traffic Sign": "P2",
    "Clogged Drain / Sewer": "P2",
    # Tier 3 → P3 (low)
    "Garbage": "P3",
    "Dead Animal": "P3",
    "Abandoned Vehicle": "P3",
    "Graffiti / Vandalism": "P3",
    "Park / Playground Damage": "P3",
}

# P0 labels that map to emergency departments
P0_DEPARTMENT_MAP = {
    "Fire Hazard": ["fire", "emergency"],
    "Car Accident": ["police", "emergency"],
    "Downed Power Line": ["electric_utility", "emergency", "fire"],
}

# Confidence threshold for P0 downgrade
P0_CONFIDENCE_THRESHOLD = 85

# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

_supported_zips = None
_city_config = None
_issue_department_map = None
_zip_code_authorities = None


def _get_data_path() -> Path:
    return Path(__file__).parent.parent / "data"


def get_supported_zips() -> List[str]:
    global _supported_zips
    if _supported_zips is None:
        try:
            with open(_get_data_path() / "supported_zips.json", "r") as f:
                data = json.load(f)
            _supported_zips = data.get("supported_zips", [])
        except Exception as e:
            logger.error(f"Failed to load supported_zips.json: {e}")
            _supported_zips = []
    return _supported_zips


def get_city_config(zip_code: str) -> Dict[str, Any]:
    global _city_config
    if _city_config is None:
        try:
            with open(_get_data_path() / "city_config.json", "r") as f:
                _city_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load city_config.json: {e}")
            _city_config = {"cities": {}, "default": {"status": "NOT_LIVE", "ai_enabled": True}}
    return _city_config.get("cities", {}).get(zip_code, _city_config.get("default", {}))


def get_department_map() -> Dict[str, Any]:
    """Load department map fresh each time for reliability."""
    try:
        with open(_get_data_path() / "issue_department_map.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load issue_department_map.json: {e}")
        return {}


def get_zip_authorities(zip_code: str) -> Dict[str, Any]:
    global _zip_code_authorities
    if _zip_code_authorities is None:
        try:
            with open(_get_data_path() / "zip_code_authorities.json", "r") as f:
                _zip_code_authorities = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load zip_code_authorities.json: {e}")
            _zip_code_authorities = {}
    return _zip_code_authorities.get(zip_code, _zip_code_authorities.get("default", {}))


def reload_all_data():
    """Force reload all cached data files."""
    global _supported_zips, _city_config, _issue_department_map, _zip_code_authorities
    _supported_zips = None
    _city_config = None
    _issue_department_map = None
    _zip_code_authorities = None
    logger.info("🔄 V5 Routing Engine: All data caches cleared")


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — ZIP GATE
# ═══════════════════════════════════════════════════════════════

def check_zip_gate(zip_code: str) -> Dict[str, Any]:
    """
    Check if zip code is in supported service area.
    Returns: { supported: bool, message: str }
    """
    supported = zip_code in get_supported_zips()
    if supported:
        return {"supported": True, "message": "Service area confirmed"}
    return {
        "supported": False,
        "message": "This location is outside our service area. Do you want to continue?",
        "flow": "OUT_OF_AREA"
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 4 — CITY STATE
# ═══════════════════════════════════════════════════════════════

def get_city_state(zip_code: str) -> Dict[str, Any]:
    """
    Determine if city is LIVE or NOT_LIVE.
    Returns: { status, ai_enabled, ai_sidecar_enabled, name }
    """
    config = get_city_config(zip_code)
    return {
        "status": config.get("status", "NOT_LIVE"),
        "ai_enabled": config.get("ai_enabled", True),
        "ai_sidecar_enabled": config.get("ai_sidecar_enabled", False),
        "ai_confidence_threshold": config.get("ai_confidence_threshold", 80),
        "name": config.get("name", "Unknown City"),
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 5 — ROUTING ENGINE
# ═══════════════════════════════════════════════════════════════

def get_severity_level(issue_label: str) -> str:
    """Get P-level severity for an issue label."""
    return ISSUE_SEVERITY_MAP.get(issue_label, "P2")


def process_issue_routing(
    issue_label: str,
    confidence: float,
    zip_code: str,
    city_status: str,
) -> List[Dict[str, Any]]:
    """
    Phase 5: Process routing for a SINGLE issue.
    
    Returns list of recipient entries:
    [{ name, email, type, issue_labels[], reason, checked, locked }]
    """
    recipients = []
    severity = get_severity_level(issue_label)
    dept_map = get_department_map()
    zip_auth = get_zip_authorities(zip_code)
    
    # ──────────────────────────────────
    # 6A: Tier 0 Safety Layer
    # ──────────────────────────────────
    if severity == "P0":
        if confidence < P0_CONFIDENCE_THRESHOLD:
            # Downgrade to P1 if confidence too low
            severity = "P1"
            logger.info(
                f"⚠️ V5 Routing: P0 downgraded to P1 for '{issue_label}' "
                f"(confidence {confidence}% < {P0_CONFIDENCE_THRESHOLD}%)"
            )
        else:
            # Add emergency departments — LOCKED, always checked
            p0_depts = P0_DEPARTMENT_MAP.get(issue_label, ["emergency", "police"])
            for dept in p0_depts:
                dept_contacts = zip_auth.get(dept, [])
                for contact in dept_contacts:
                    recipients.append({
                        "name": contact.get("name", dept),
                        "email": contact.get("email", ""),
                        "type": contact.get("type", dept),
                        "issue_labels": [issue_label],
                        "reason": f"P0_emergency_{issue_label.lower().replace(' ', '_')}",
                        "checked": True,
                        "locked": True,
                        "severity": "P0",
                    })
            logger.info(f"🔴 V5 Routing: P0 LOCKED recipients added for '{issue_label}'")
    
    # ──────────────────────────────────
    # 6B: Department Mapping
    # ──────────────────────────────────
    import re
    # Normalize: "Fallen / Hazardous Tree" → "fallen_hazardous_tree"
    issue_key = re.sub(r'[_\s]+', '_', issue_label.lower().strip().replace("/", "_")).strip("_")
    
    # Try multiple key formats for maximum match rate
    mapped_depts = dept_map.get(issue_key)
    if not mapped_depts:
        # Try first word only: "fallen_hazardous_tree" → "fallen_tree"
        words = issue_key.split("_")
        if len(words) >= 2:
            short_key = f"{words[0]}_{words[-1]}"
            mapped_depts = dept_map.get(short_key)
    if not mapped_depts:
        # Try exact label lowered with underscores
        alt_key = issue_label.lower().strip().replace(" ", "_")
        mapped_depts = dept_map.get(alt_key)
    
    if mapped_depts:
        # CASE A: Mapping found
        for dept in mapped_depts:
            # Skip if already added as P0 locked
            if any(r["type"] == dept and r["locked"] for r in recipients):
                continue
            dept_contacts = zip_auth.get(dept, [])
            for contact in dept_contacts:
                recipients.append({
                    "name": contact.get("name", dept),
                    "email": contact.get("email", ""),
                    "type": contact.get("type", dept),
                    "issue_labels": [issue_label],
                    "reason": "mapped_department",
                    "checked": True,
                    "locked": False,
                    "severity": severity,
                })
    else:
        # CASE B: Mapping NOT found → route to MomntumAI Crew
        recipients.append({
            "name": "MomntumAI Crew",
            "email": "eaiser@momntumai.com",
            "type": "internal_review",
            "issue_labels": [issue_label],
            "reason": "unknown_label",
            "checked": True,
            "locked": False,
            "severity": severity,
        })
        logger.info(f"🟡 V5 Routing: Unknown label '{issue_label}' → MomntumAI Crew")
    
    return recipients


# ═══════════════════════════════════════════════════════════════
# PHASE 6 — BUILD RECIPIENT LIST (with dedup)
# ═══════════════════════════════════════════════════════════════

def build_recipient_list(
    issues: List[Dict[str, Any]],
    zip_code: str,
    city_status: str = "LIVE",
    is_out_of_area: bool = False,
) -> Dict[str, Any]:
    """
    Build the final recipient list from all issues.
    
    Args:
        issues: [{ label, confidence, severity, observations }]
        zip_code: User's zip code
        city_status: LIVE or NOT_LIVE
        is_out_of_area: If True, route everything to MomntumAI only
    
    Returns:
        {
            recipients: [{ name, email, type, issue_labels[], reason, checked, locked, severity }],
            has_p0: bool,
            advisory_text: str | None,
            city_status: str,
            is_out_of_area: bool,
        }
    """
    # Phase 15: Out of area flow
    if is_out_of_area:
        return {
            "recipients": [{
                "name": "MomntumAI Crew",
                "email": "eaiser@momntumai.com",
                "type": "internal_review",
                "issue_labels": [i.get("label", "Unknown") for i in issues],
                "reason": "out_of_area",
                "checked": True,
                "locked": False,
                "severity": "P2",
            }],
            "has_p0": False,
            "advisory_text": None,
            "city_status": city_status,
            "is_out_of_area": True,
        }
    
    # Phase 4: Draft flow for NOT_LIVE cities
    if city_status == "NOT_LIVE":
        return {
            "recipients": [{
                "name": "MomntumAI Crew (Draft)",
                "email": "eaiser@momntumai.com",
                "type": "internal_draft",
                "issue_labels": [i.get("label", "Unknown") for i in issues],
                "reason": "city_not_live",
                "checked": True,
                "locked": False,
                "severity": "P2",
            }],
            "has_p0": False,
            "advisory_text": None,
            "city_status": city_status,
            "is_out_of_area": False,
        }
    
    # Process each issue through routing engine
    all_recipients = []
    has_p0 = False
    advisory_text = None
    
    for issue in issues:
        label = issue.get("label", "Unknown")
        confidence = issue.get("confidence", 0)
        severity = get_severity_level(label)
        
        if severity == "P0" and confidence >= P0_CONFIDENCE_THRESHOLD:
            has_p0 = True
        
        issue_recipients = process_issue_routing(
            issue_label=label,
            confidence=confidence,
            zip_code=zip_code,
            city_status=city_status,
        )
        all_recipients.extend(issue_recipients)
    
    # ──────────────────────────────────
    # 6C: Deduplicate by department
    # ──────────────────────────────────
    deduped = {}
    for r in all_recipients:
        key = f"{r['email']}_{r['type']}"
        if key in deduped:
            # Merge issue labels
            existing = deduped[key]
            for label in r["issue_labels"]:
                if label not in existing["issue_labels"]:
                    existing["issue_labels"].append(label)
            # Keep locked if ANY entry is locked
            if r["locked"]:
                existing["locked"] = True
                existing["checked"] = True
            # Keep highest severity
            if SEVERITY_MAP.get(r["severity"], {}).get("priority_ordinal", 0) > \
               SEVERITY_MAP.get(existing["severity"], {}).get("priority_ordinal", 0):
                existing["severity"] = r["severity"]
        else:
            deduped[key] = r.copy()
    
    recipients = list(deduped.values())
    
    # Build advisory text for P0
    if has_p0:
        advisory_text = "⚠️ This report contains emergency-level issues. Call 911 immediately if there is immediate danger."
    
    return {
        "recipients": recipients,
        "has_p0": has_p0,
        "advisory_text": advisory_text,
        "city_status": city_status,
        "is_out_of_area": False,
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 9 — HARD GUARDRAIL
# ═══════════════════════════════════════════════════════════════

def apply_hard_guardrail(recipients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    HARD RULE: If all NON-LOCKED recipients are unchecked,
    auto-add MomntumAI Crew as checked.
    
    Returns updated recipients list.
    """
    has_any_checked = any(r["checked"] for r in recipients)
    
    if not has_any_checked:
        # All unchecked → add MomntumAI Crew
        recipients.append({
            "name": "MomntumAI Crew",
            "email": "eaiser@momntumai.com",
            "type": "internal_fallback",
            "issue_labels": [],
            "reason": "hard_guardrail_zero_recipients",
            "checked": True,
            "locked": False,
            "severity": "P2",
        })
        logger.warning("🔴 V5 Hard Guardrail: All recipients unchecked → Added MomntumAI Crew")
    
    return recipients


# ═══════════════════════════════════════════════════════════════
# PHASE 10 — PERSISTENCE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_consent_log(
    issue_id: str,
    recipients: List[Dict[str, Any]],
    user_id: str,
) -> Dict[str, Any]:
    """Build consent log document for DB persistence."""
    return {
        "issue_id": issue_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "entries": [
            {
                "recipient_name": r["name"],
                "recipient_email": r["email"],
                "recipient_type": r["type"],
                "checked": r["checked"],
                "locked": r["locked"],
                "reason": "user_tagged" if r["checked"] else "user_untagged",
            }
            for r in recipients
        ],
    }


def build_report_routings(
    issue_id: str,
    recipients: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build report_routings documents — one row per checked recipient."""
    return [
        {
            "issue_id": issue_id,
            "recipient_name": r["name"],
            "recipient_email": r["email"],
            "recipient_type": r["type"],
            "issue_labels": r["issue_labels"],
            "severity": r["severity"],
            "locked": r["locked"],
            "routed_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        for r in recipients if r["checked"]
    ]


def build_report_ownership(
    issue_id: str,
    recipients: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build report_ownership documents — issue × department mapping."""
    entries = []
    for r in recipients:
        if r["checked"]:
            for label in r["issue_labels"]:
                entries.append({
                    "issue_id": issue_id,
                    "issue_label": label,
                    "department": r["type"],
                    "recipient_name": r["name"],
                    "assigned_at": datetime.utcnow().isoformat(),
                    "status": "assigned",
                })
    return entries


def build_audit_log(
    issue_id: str,
    action: str,
    user_id: str,
    before: Any = None,
    after: Any = None,
    details: str = "",
) -> Dict[str, Any]:
    """Build full audit log entry with before/after diff."""
    return {
        "issue_id": issue_id,
        "action": action,
        "user_id": user_id,
        "before": before,
        "after": after,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 9 — AI SIDE-CAR (Safe Mode)
# ═══════════════════════════════════════════════════════════════

def generate_ai_suggestions(
    issues: List[Dict[str, Any]],
    current_recipients: List[Dict[str, Any]],
    city_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    AI Side-Car: Generate additional routing suggestions.
    
    Constraints:
    - Cannot override existing recipients
    - Cannot suggest Tier 0
    - Must pass confidence threshold (>=80)
    - Only runs if city is LIVE and ai_sidecar_enabled
    """
    if not city_config.get("ai_sidecar_enabled", False):
        return []
    
    threshold = city_config.get("ai_confidence_threshold", 80)
    suggestions = []
    
    # Current recipient types for dedup
    existing_types = {r["type"] for r in current_recipients}
    
    for issue in issues:
        severity = get_severity_level(issue.get("label", ""))
        confidence = issue.get("confidence", 0)
        
        # Skip P0 (cannot suggest emergency)
        if severity == "P0":
            continue
        # Skip low confidence
        if confidence < threshold:
            continue
        
        # Check if there are unmapped departments that could be relevant
        # This is a placeholder — real AI logic would go here
        pass
    
    return suggestions


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_routing_pipeline(
    issues: List[Dict[str, Any]],
    zip_code: str,
    user_continued_out_of_area: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestrator: Runs the complete V5 routing pipeline.
    
    Args:
        issues: AI-detected issues [{ label, confidence, severity, observations }]
        zip_code: User's zip code
        user_continued_out_of_area: If user chose to continue despite out-of-area
    
    Returns:
        Complete routing result with recipients, advisory, metadata
    """
    # Phase 2: ZIP Gate
    zip_gate = check_zip_gate(zip_code)
    is_out_of_area = not zip_gate["supported"]
    
    if is_out_of_area and not user_continued_out_of_area:
        return {
            "phase": "zip_gate",
            "zip_gate": zip_gate,
            "recipients": [],
            "requires_user_confirmation": True,
        }
    
    # Phase 4: City State
    city_state = get_city_state(zip_code)
    
    # Phase 5-6: Build recipient list
    result = build_recipient_list(
        issues=issues,
        zip_code=zip_code,
        city_status=city_state["status"],
        is_out_of_area=is_out_of_area and user_continued_out_of_area,
    )
    
    # Phase 9: AI Side-Car (if applicable)
    city_cfg = get_city_config(zip_code)
    ai_suggestions = generate_ai_suggestions(issues, result["recipients"], city_cfg)
    
    return {
        "phase": "recipient_list_ready",
        "zip_gate": zip_gate,
        "city_state": city_state,
        "recipients": result["recipients"],
        "has_p0": result["has_p0"],
        "advisory_text": result["advisory_text"],
        "is_out_of_area": result["is_out_of_area"],
        "ai_suggestions": ai_suggestions,
        "requires_user_confirmation": False,
        "total_recipients": len(result["recipients"]),
        "locked_count": sum(1 for r in result["recipients"] if r["locked"]),
        "toggleable_count": sum(1 for r in result["recipients"] if not r["locked"]),
    }
