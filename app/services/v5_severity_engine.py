"""
EAiSER V5 Severity Engine
===========================
Deterministic 13-factor severity scoring system.
Replaces V4's tier-based priority with a risk-factor rubric.

Architecture:
  - 7 Core Boolean Factors (presence_of_hazard, structural_risk, etc.)
  - 1 Hard Rule (accessibility_impact → minimum High)
  - 5 Modifier Factors (active_vs_latent, road_speed, scope, vulnerable_pop, time_sensitivity)
  - 8 Tier 0 Emergency Categories (fire, gas leak, flooding, power line, accident, collapse, hazmat, distress)
  - Issue ID format: r_{8-char-hash}_i{NN}

If this engine and the AI disagree, THIS ENGINE WINS. Always.
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Severity levels (ordered)
SEVERITY_LEVELS = ["Low", "Medium", "High", "Emergency"]

SEVERITY_ORDINAL: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Emergency": 4,
}

# 8 Tier 0 Emergency Categories
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

# 16 Known Issue Categories (non-emergency)
KNOWN_CATEGORIES = {
    "pothole", "garbage", "broken_streetlight", "road_damage",
    "graffiti_vandalism", "damaged_sidewalk", "damaged_traffic_sign",
    "fallen_tree", "clogged_drain", "park_playground_damage",
    "broken_traffic_signal", "abandoned_vehicle", "dead_animal",
    "water_leakage", "open_manhole", "flooding",
}

ALL_KNOWN_LABELS = TIER_0_CATEGORIES | KNOWN_CATEGORIES

# Unified 911 Banner Template (locked — do not modify without legal review)
EMERGENCY_BANNER_TEMPLATE = (
    "Call 911 if this is an emergency. This image may show {issue_type}. "
    "EAiSER is not monitored in real time. Call 911."
)

EMERGENCY_BANNER_FULL = (
    "⚠️ This may need 911\n\n"
    "Some of what you reported looks like it could be an emergency. "
    "EAiSER is not a 911 service and is not monitored 24/7. "
    "If anyone is in danger, or there is any reason to reach out to an "
    "emergency line, please call 911.\n\n"
    "You can still submit this report and we'll work to route it to the "
    "right departments. For emergencies, call 911."
)

# Analysis status values (7 possible)
ANALYSIS_STATUS_VALUES = [
    "clean",              # All issues known, no review needed
    "has_unknown",        # At least one unknown issue
    "low_confidence",     # Primary issue confidence < 0.50
    "emergency_detected", # Tier 0 issue found
    "no_issue_found",     # AI found nothing
    "image_rejected",     # Fake/non-civic image
    "error",              # AI call failed
]

# Max issues per report
MAX_ISSUES_PER_REPORT = 5

# Label normalization aliases
LABEL_ALIASES: Dict[str, str] = {
    "tree_fallen": "fallen_tree",
    "tree_down": "fallen_tree",
    "fallen___hazardous_tree": "fallen_tree",
    "fallen_hazardous_tree": "fallen_tree",
    "hazardous_tree": "fallen_tree",
    "graffiti": "graffiti_vandalism",
    "vandalism": "graffiti_vandalism",
    "clogged_drain___sewer": "clogged_drain",
    "clogged_drain_sewer": "clogged_drain",
    "sewer": "clogged_drain",
    "open___damaged_manhole": "open_manhole",
    "open_damaged_manhole": "open_manhole",
    "damaged_manhole": "open_manhole",
    "flooding___standing_water": "flooding",
    "flooding_standing_water": "flooding",
    "standing_water": "flooding",
    "water_leak": "water_leakage",
    "park_damage": "park_playground_damage",
    "playground_damage": "park_playground_damage",
    "traffic_signal": "broken_traffic_signal",
    "power_line": "downed_power_line",
    "vehicle_damage": "car_accident",
    "tree_on_vehicle": "car_accident",
    "tree_on_car": "car_accident",
    "illegal_dumping": "garbage",
}


# ═══════════════════════════════════════════════════════════════
# LABEL NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_label(label: str) -> str:
    """Normalize an issue label to canonical snake_case form."""
    normalized = label.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    # Remove doubled underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    # Check aliases
    return LABEL_ALIASES.get(normalized, normalized)


def is_known_label(label: str) -> bool:
    """Check if a normalized label is in the known set."""
    return normalize_label(label) in ALL_KNOWN_LABELS


def is_tier_0(label: str) -> bool:
    """Check if a normalized label is a Tier 0 emergency."""
    return normalize_label(label) in TIER_0_CATEGORIES


# ═══════════════════════════════════════════════════════════════
# ISSUE ID GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_report_hash(image_bytes: bytes = None, timestamp: str = None) -> str:
    """Generate stable 8-char hash for report ID prefix."""
    if image_bytes:
        h = hashlib.sha256(image_bytes[:4096]).hexdigest()[:8]
    elif timestamp:
        h = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
    else:
        h = hashlib.sha256(datetime.utcnow().isoformat().encode()).hexdigest()[:8]
    return h


def generate_issue_id(report_hash: str, issue_index: int) -> str:
    """Generate V5 issue ID: r_{hash}_i{NN}"""
    return f"r_{report_hash}_i{issue_index:02d}"


# ═══════════════════════════════════════════════════════════════
# 13-FACTOR RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════

# The 13 risk factors:
#   CORE (7 booleans):
#     1. presence_of_hazard         — Is there an active danger to people?
#     2. structural_risk            — Could something collapse / further break?
#     3. obstruction_of_passage     — Blocks road, sidewalk, or path?
#     4. utility_service_impact     — Affects water, power, gas, sewer?
#     5. environmental_contamination — Polluting soil, water, air?
#     6. visibility_or_signage_issue — Missing/damaged signs or poor visibility?
#     7. repeat_or_worsening        — Known recurring or escalating issue?
#   HARD RULE (1):
#     8. accessibility_impact       — ADA non-compliant → minimum High
#   MODIFIERS (5):
#     9.  active_vs_latent          — Active hazard vs latent risk
#     10. road_speed_context        — High-speed road increases risk
#     11. scope                     — single-point vs area-wide
#     12. vulnerable_population     — Near school, hospital, senior center
#     13. time_sensitivity          — Gets worse with time?

DEFAULT_RISK_FACTORS = {
    "presence_of_hazard": False,
    "structural_risk": False,
    "obstruction_of_passage": False,
    "utility_service_impact": False,
    "environmental_contamination": False,
    "visibility_or_signage_issue": False,
    "repeat_or_worsening": False,
    "accessibility_impact": False,
    "active_vs_latent": "latent",   # "active" or "latent"
    "road_speed_context": "low",    # "low", "medium", "high"
    "scope": "single_point",        # "single_point" or "area_wide"
    "vulnerable_population": False,
    "time_sensitivity": "stable",   # "stable", "degrading", "urgent"
}


def compute_severity_from_factors(
    label: str,
    risk_factors: Dict[str, Any],
) -> Tuple[str, bool, List[str]]:
    """
    Deterministic severity calculation from 13 risk factors.

    Returns: (severity, is_emergency, explanation_list)
    """
    norm = normalize_label(label)
    explanations = []

    # RULE 0: Tier 0 categories are ALWAYS Emergency
    if norm in TIER_0_CATEGORIES:
        explanations.append(f"Tier 0 category '{norm}' → Emergency (locked)")
        return "Emergency", True, explanations

    # Count core boolean factors that are True
    core_booleans = [
        "presence_of_hazard", "structural_risk", "obstruction_of_passage",
        "utility_service_impact", "environmental_contamination",
        "visibility_or_signage_issue", "repeat_or_worsening"
    ]
    true_count = sum(1 for k in core_booleans if risk_factors.get(k, False))

    # HARD RULE: ADA accessibility_impact → minimum High
    ada_triggered = risk_factors.get("accessibility_impact", False)
    if ada_triggered:
        explanations.append("ADA accessibility_impact=true → minimum High")

    # BASE SEVERITY from core boolean count
    if true_count >= 5:
        base_severity = "High"
        explanations.append(f"{true_count}/7 core risk factors active → High")
    elif true_count >= 3:
        base_severity = "Medium"
        explanations.append(f"{true_count}/7 core risk factors active → Medium")
    elif true_count >= 1:
        base_severity = "Low"
        explanations.append(f"{true_count}/7 core risk factors active → Low")
    else:
        base_severity = "Low"
        explanations.append("No core risk factors active → Low")

    # MODIFIER ADJUSTMENTS
    modifier_boost = 0

    # Modifier 1: active_vs_latent
    if risk_factors.get("active_vs_latent") == "active":
        modifier_boost += 1
        explanations.append("Active hazard (not latent) → +1")

    # Modifier 2: road_speed_context
    if risk_factors.get("road_speed_context") == "high":
        modifier_boost += 1
        explanations.append("High-speed road context → +1")

    # Modifier 3: scope
    if risk_factors.get("scope") == "area_wide":
        modifier_boost += 1
        explanations.append("Area-wide scope → +1")

    # Modifier 4: vulnerable_population
    if risk_factors.get("vulnerable_population", False):
        modifier_boost += 1
        explanations.append("Near vulnerable population → +1")

    # Modifier 5: time_sensitivity
    ts = risk_factors.get("time_sensitivity", "stable")
    if ts == "urgent":
        modifier_boost += 2
        explanations.append("Urgent time sensitivity → +2")
    elif ts == "degrading":
        modifier_boost += 1
        explanations.append("Degrading time sensitivity → +1")

    # Apply modifier boosts to base severity
    severity_ordinal = SEVERITY_ORDINAL.get(base_severity, 1) + modifier_boost
    # Cap at High for non-Tier-0 (only Tier 0 can be Emergency)
    severity_ordinal = min(severity_ordinal, SEVERITY_ORDINAL["High"])

    # ADA hard rule enforcement
    if ada_triggered:
        severity_ordinal = max(severity_ordinal, SEVERITY_ORDINAL["High"])

    # Reverse lookup
    final_severity = "Low"
    for sev, ordinal in SEVERITY_ORDINAL.items():
        if ordinal == severity_ordinal:
            final_severity = sev
            break
    # If ordinal exceeds High but not Tier 0, cap at High
    if severity_ordinal >= SEVERITY_ORDINAL["High"] and not is_tier_0(label):
        final_severity = "High"

    return final_severity, False, explanations


# ═══════════════════════════════════════════════════════════════
# REPORT-LEVEL SEVERITY COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_report_severity(
    issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute overall report severity from all issues.

    Returns:
        {
            "final_priority": str,        # Emergency/High/Medium/Low
            "has_emergency": bool,
            "emergency_categories": list,
            "primary_issue_index": int,
            "analysis_status": str,
            "emergency_banner": str|None,
            "emergency_banner_full": str|None,
        }
    """
    if not issues and not unknown_issues:
        return {
            "final_priority": "Low",
            "has_emergency": False,
            "emergency_categories": [],
            "primary_issue_index": 0,
            "analysis_status": "no_issue_found",
            "emergency_banner": None,
            "emergency_banner_full": None,
        }

    unknown_issues = unknown_issues or []
    max_ordinal = 0
    has_emergency = False
    emergency_cats = []
    primary_idx = 0
    primary_ordinal = 0

    # Evaluate each issue
    for idx, issue in enumerate(issues):
        label = issue.get("label", issue.get("issue", ""))
        norm = normalize_label(label)
        risk_factors = issue.get("risk_factors", DEFAULT_RISK_FACTORS.copy())

        severity, is_emg, _ = compute_severity_from_factors(norm, risk_factors)
        sev_ord = SEVERITY_ORDINAL.get(severity, 1)

        if is_emg:
            has_emergency = True
            emergency_cats.append(norm)
            sev_ord = SEVERITY_ORDINAL["Emergency"]

        max_ordinal = max(max_ordinal, sev_ord)

        # Primary = highest severity; tiebreak by life-impact then array order
        life_impact = risk_factors.get("presence_of_hazard", False)
        current_primary_life = False
        if primary_idx < len(issues):
            pf = issues[primary_idx].get("risk_factors", {})
            current_primary_life = pf.get("presence_of_hazard", False)

        if sev_ord > primary_ordinal or (sev_ord == primary_ordinal and life_impact and not current_primary_life):
            primary_ordinal = sev_ord
            primary_idx = idx

    # Unknown issues contribute severity but can't be Emergency
    for u in unknown_issues:
        sev = u.get("severity", "Medium")
        sev_ord = min(SEVERITY_ORDINAL.get(sev, 2), SEVERITY_ORDINAL["High"])
        max_ordinal = max(max_ordinal, sev_ord)

    # Reverse lookup
    final_priority = "Low"
    for p, o in SEVERITY_ORDINAL.items():
        if o == max_ordinal:
            final_priority = p
            break

    # Determine analysis status
    has_unknown = len(unknown_issues) > 0
    if has_emergency:
        analysis_status = "emergency_detected"
    elif has_unknown:
        analysis_status = "has_unknown"
    else:
        analysis_status = "clean"

    # Emergency banner
    banner = None
    banner_full = None
    if has_emergency and emergency_cats:
        # Use highest-precedence Tier 0 category
        precedence = [
            "downed_power_line", "fire_hazard", "fire", "major_gas_leak",
            "active_flooding", "structural_collapse", "hazmat_spill",
            "car_accident", "person_in_distress"
        ]
        primary_emg = emergency_cats[0]
        for p in precedence:
            if p in emergency_cats:
                primary_emg = p
                break
        display_name = primary_emg.replace("_", " ").title()
        banner = EMERGENCY_BANNER_TEMPLATE.format(issue_type=display_name)
        banner_full = EMERGENCY_BANNER_FULL

    return {
        "final_priority": final_priority,
        "has_emergency": has_emergency,
        "emergency_categories": emergency_cats,
        "primary_issue_index": primary_idx,
        "analysis_status": analysis_status,
        "emergency_banner": banner,
        "emergency_banner_full": banner_full,
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE TRUNCATION (5-issue cap)
# ═══════════════════════════════════════════════════════════════

def truncate_issues(
    issues: List[Dict[str, Any]],
    max_count: int = MAX_ISSUES_PER_REPORT,
) -> Tuple[List[Dict[str, Any]], bool, int]:
    """
    Enforce 5-issue cap with deterministic tiebreak.
    Sort: severity desc → life-impact → array order.
    Returns: (truncated_list, was_truncated, dropped_count)
    """
    if len(issues) <= max_count:
        return issues, False, 0

    def sort_key(item):
        label = normalize_label(item.get("label", item.get("issue", "")))
        rf = item.get("risk_factors", {})
        sev_ord = SEVERITY_ORDINAL.get(item.get("computed_severity", "Low"), 1)
        if is_tier_0(label):
            sev_ord = SEVERITY_ORDINAL["Emergency"]
        life = 1 if rf.get("presence_of_hazard", False) else 0
        return (-sev_ord, -life)

    sorted_issues = sorted(issues, key=sort_key)
    dropped = len(issues) - max_count

    logger.info(f"📋 V5: Truncated {len(issues)} issues to {max_count} (dropped {dropped})")
    return sorted_issues[:max_count], True, dropped


# ═══════════════════════════════════════════════════════════════
# V5 REPORT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_v5_report_envelope(
    issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]],
    report_hash: str,
    scene_description: str = "",
    image_count: int = 1,
) -> Dict[str, Any]:
    """
    Build the complete V5 report envelope.
    """
    # Assign issue IDs
    for idx, issue in enumerate(issues):
        issue["issue_id"] = generate_issue_id(report_hash, idx + 1)
        # Compute severity from risk factors
        label = issue.get("label", issue.get("issue", ""))
        rf = issue.get("risk_factors", DEFAULT_RISK_FACTORS.copy())
        severity, is_emg, explanations = compute_severity_from_factors(label, rf)
        issue["computed_severity"] = severity
        issue["is_emergency"] = is_emg
        issue["severity_explanations"] = explanations

    # Truncate
    issues, truncated, dropped = truncate_issues(issues)

    # Compute report-level severity
    report_sev = compute_report_severity(issues, unknown_issues)

    # Determine primary issue
    primary_idx = report_sev["primary_issue_index"]
    primary_issue_id = issues[primary_idx]["issue_id"] if issues else None

    return {
        "schema_version": "v5" if image_count == 1 else "v5-multi",
        "simi_version": "v5.0.0",
        "report_hash": report_hash,
        "analysis_status": report_sev["analysis_status"],
        "final_priority": report_sev["final_priority"],
        "primary_issue_id": primary_issue_id,
        "truncated": truncated,
        "truncated_count": dropped,
        "report_summary": scene_description,
        "report_status": "ai_generated",
        "summary_freshness": "current",
        "requires_post_action_review": report_sev["analysis_status"] in (
            "has_unknown", "low_confidence", "emergency_detected"
        ),
        "issues": issues,
        "unknown_issues": unknown_issues,
        "edit_log": [],
        "emergency": {
            "has_emergency": report_sev["has_emergency"],
            "categories": report_sev["emergency_categories"],
            "banner": report_sev["emergency_banner"],
            "banner_full": report_sev["emergency_banner_full"],
        },
        "created_at": datetime.utcnow().isoformat(),
    }
