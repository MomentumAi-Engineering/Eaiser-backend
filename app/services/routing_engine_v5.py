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

# V5 Severity levels (replaces P0-P3)
SEVERITY_MAP = {
    "Emergency": {"label": "Emergency", "description": "Life/Safety — 911", "priority_ordinal": 4},
    "High": {"label": "High", "description": "Same-day review", "priority_ordinal": 3},
    "Medium": {"label": "Medium", "description": "This week", "priority_ordinal": 2},
    "Low": {"label": "Low", "description": "Routine maintenance", "priority_ordinal": 1},
}

# Legacy P-level aliases for backward compat
P_LEVEL_TO_SEVERITY = {"P0": "Emergency", "P1": "High", "P2": "Medium", "P3": "Low"}

# 8 Tier 0 categories (V5 spec)
TIER_0_CATEGORIES = {
    "fire_hazard", "fire",
    "major_gas_leak",
    "active_flooding",
    "downed_power_line",
    "car_accident",
    "structural_collapse",
    "hazmat_spill",
    "person_in_distress",
}

# Confidence threshold for 911 banner (float 0.0-1.0)
TIER_0_CONFIDENCE_THRESHOLD = 0.85

# Unified 911 banner (locked — do not modify without legal review)
EMERGENCY_BANNER = (
    "⚠️ This may need 911\n\n"
    "Some of what you reported looks like it could be an emergency. "
    "EAiSER is not a 911 service and is not monitored 24/7. "
    "If anyone is in danger, or there is any reason to reach out to an "
    "emergency line, please call 911.\n\n"
    "You can still submit this report and we'll work to route it to the "
    "right departments. For emergencies, call 911."
)

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
# PHASE 5 — ROUTING ENGINE (V5: severity-aware)
# ═══════════════════════════════════════════════════════════════

def _normalize_issue_key(label: str) -> str:
    """Normalize issue label to snake_case for department map lookup."""
    import re
    key = re.sub(r'[_\s]+', '_', label.lower().strip().replace("/", "_").replace("-", "_")).strip("_")
    # Remove doubled underscores
    while "__" in key:
        key = key.replace("__", "_")
    return key


def _resolve_aliases(issue_key: str, dept_map: Dict[str, Any]) -> str:
    """Resolve issue aliases from the department map."""
    aliases = dept_map.get("_aliases", {})
    if issue_key in aliases:
        return aliases[issue_key]
    return issue_key


def _get_severity_key(severity: str) -> str:
    """Convert severity to department map key ('low', 'medium', 'high', 'emergency')."""
    if not severity:
        return "medium"
    s = severity.strip().lower()
    # Handle legacy P-levels
    if s.startswith("p"):
        mapped = P_LEVEL_TO_SEVERITY.get(severity.upper(), "Medium")
        return mapped.lower()
    # Handle Tier-based severity from AI
    if "tier" in s or "tier_" in s:
        if "0" in s:
            return "emergency"
        elif "1" in s:
            return "high"
        elif "2" in s:
            return "medium"
        elif "3" in s:
            return "low"
        return "medium"
    # Handle direct severity words
    if s in ("emergency", "critical"):
        return "emergency"
    if s in ("high", "urgent"):
        return "high"
    if s in ("medium", "moderate"):
        return "medium"
    if s in ("low", "minor"):
        return "low"
    return "medium"  # safe default


def process_issue_routing(
    issue_label: str,
    confidence: float,
    severity: str,
    zip_code: str,
    city_status: str,
    lat: float = None,
    lng: float = None,
) -> List[Dict[str, Any]]:
    """
    Phase 5: Process routing for a SINGLE issue.
    V5: Uses severity-aware department map with per-level recipient lists.
    
    Args:
        issue_label: The issue type label
        confidence: Float 0.0-1.0
        severity: "Emergency", "High", "Medium", or "Low"
        zip_code: User's zip code
        city_status: LIVE or NOT_LIVE
        lat/lng: Optional coordinates for location-conditional routing
    
    Returns list of recipient entries:
    [{ name, email, type, issue_labels[], reason, checked, locked, severity }]
    """
    recipients = []
    dept_map = get_department_map()
    zip_auth = get_zip_authorities(zip_code)
    
    # Normalize issue key and resolve aliases
    issue_key = _normalize_issue_key(issue_label)
    issue_key = _resolve_aliases(issue_key, dept_map)
    
    # Normalize severity to key
    sev_key = _get_severity_key(severity)
    
    # Check if Tier 0
    is_tier_0 = issue_key in TIER_0_CATEGORIES
    if is_tier_0:
        sev_key = "emergency"  # Tier 0 always routes at Emergency
    
    # ──────────────────────────────────
    # 6A: Tier 0 Safety Layer
    # ──────────────────────────────────
    if is_tier_0:
        if confidence < TIER_0_CONFIDENCE_THRESHOLD:
            # Don't fire 911 banner, but still route to departments
            logger.info(
                f"⚠️ V5 Routing: Tier 0 '{issue_key}' confidence {confidence:.2f} "
                f"< {TIER_0_CONFIDENCE_THRESHOLD} — 911 banner suppressed, ops review required"
            )
    
    # ──────────────────────────────────
    # 6B: Severity-Aware Department Mapping
    # ──────────────────────────────────
    issue_config = dept_map.get(issue_key)
    
    # Try alias resolution if not found
    if not issue_config:
        # Try common alternatives
        alt_keys = [issue_key]
        words = issue_key.split("_")
        if len(words) >= 2:
            alt_keys.append(f"{words[0]}_{words[-1]}")
        for alt in alt_keys:
            resolved = _resolve_aliases(alt, dept_map)
            if resolved in dept_map and not resolved.startswith("_"):
                issue_config = dept_map[resolved]
                break
    
    if not issue_config or isinstance(issue_config, str):
        # Use fallback
        issue_config = dept_map.get("_fallback", {
            "low": ["general"], "medium": ["general"],
            "high": ["general"], "emergency": ["general", "county_ema"]
        })
    
    # Get severity-specific department list
    if isinstance(issue_config, dict):
        mapped_depts = issue_config.get(sev_key)
        # If severity level returns null (Tier 0 at non-emergency), use emergency
        if mapped_depts is None:
            mapped_depts = issue_config.get("emergency", issue_config.get("high", ["general"]))
        
        # Apply location rules (SR 96/100 → add TDOT)
        location_rules = issue_config.get("location_rules", [])
        for rule in location_rules:
            if rule.get("condition") == "on_state_route" and lat and lng:
                # TODO: GeoJSON proximity check (30-foot buffer)
                # For now, log that location rules exist
                logger.info(f"📍 Location rule exists for {issue_key}: {rule.get('routes')}")
    elif isinstance(issue_config, list):
        # Legacy flat format: just a list of departments
        mapped_depts = issue_config
    else:
        mapped_depts = ["general"]
    
    if mapped_depts:
        for dept in mapped_depts:
            dept_contacts = zip_auth.get(dept, [])
            if not dept_contacts:
                logger.warning(f"⚠️ No contacts found for dept '{dept}' in ZIP {zip_code}")
                continue
            for contact in dept_contacts:
                is_locked = is_tier_0  # Tier 0 recipients are always locked
                recipients.append({
                    "name": contact.get("name", dept),
                    "email": contact.get("email", ""),
                    "type": contact.get("type", dept),
                    "issue_labels": [issue_label],
                    "reason": f"tier_0_emergency_{issue_key}" if is_tier_0 else "severity_based_routing",
                    "checked": True,
                    "locked": is_locked,
                    "severity": "Emergency" if is_tier_0 else severity,
                    "contact_name": contact.get("contact_name", ""),
                    "phone": contact.get("phone", ""),
                })
        if is_tier_0:
            logger.info(f"🔴 V5 Routing: Tier 0 LOCKED recipients for '{issue_key}': {[d for d in mapped_depts]}")
        else:
            logger.info(f"📧 V5 Routing: {sev_key} recipients for '{issue_key}': {[d for d in mapped_depts]}")
    else:
        # No mapping found → route to MomntumAI Ops Queue
        recipients.append({
            "name": "MomntumAI Ops Queue",
            "email": "ops-queue@momntumai-routing.dev",
            "type": "internal_review",
            "issue_labels": [issue_label],
            "reason": "unknown_label",
            "checked": True,
            "locked": False,
            "severity": severity,
        })
        logger.info(f"🟡 V5 Routing: Unknown label '{issue_key}' → Ops Queue")
    
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
    has_emergency = False
    advisory_text = None
    emergency_categories = []
    
    for issue in issues:
        label = issue.get("label", "Unknown")
        confidence = issue.get("confidence", 0)
        # Normalize confidence: convert percentage to float if needed
        if isinstance(confidence, (int, float)) and confidence > 1.0:
            confidence = confidence / 100.0
        severity = issue.get("severity", issue.get("computed_severity", "Medium"))
        # Handle legacy P-levels
        if isinstance(severity, str) and severity.startswith("P"):
            severity = P_LEVEL_TO_SEVERITY.get(severity, "Medium")
        
        issue_key = _normalize_issue_key(label)
        if issue_key in TIER_0_CATEGORIES and confidence >= TIER_0_CONFIDENCE_THRESHOLD:
            has_emergency = True
            emergency_categories.append(issue_key)
        
        issue_recipients = process_issue_routing(
            issue_label=label,
            confidence=confidence,
            severity=severity,
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
            existing = deduped[key]
            for lbl in r["issue_labels"]:
                if lbl not in existing["issue_labels"]:
                    existing["issue_labels"].append(lbl)
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
    
    # Build advisory text for emergency
    if has_emergency:
        advisory_text = EMERGENCY_BANNER
    
    return {
        "recipients": recipients,
        "has_p0": has_emergency,  # backward compat key name
        "has_emergency": has_emergency,
        "emergency_categories": emergency_categories,
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
                "source": r.get("source", "ai_recommended"),
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
            "source": r.get("source", "ai_recommended"),
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
            labels = r["issue_labels"] or [""]
            for label in labels:
                entries.append({
                    "issue_id": issue_id,
                    "issue_label": label,
                    "department": r["type"],
                    "recipient_name": r["name"],
                    "source": r.get("source", "ai_recommended"),
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
