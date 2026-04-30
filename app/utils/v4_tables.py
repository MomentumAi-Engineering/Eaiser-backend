"""
EAiSER V4 Authoritative Tables & Reconciler
============================================
Deterministic tables and logic that override AI output.
Based on: EAiSER-V4-Full-Logic-Explained.md

These tables are the ONLY source of truth for:
- Issue → Tier mapping
- Tier → Priority mapping
- Emergency advisory text (liability-reviewed)
- Scenario determination (A-F)
- Reconciliation (consistency enforcement)

If a table and the AI disagree, the TABLE WINS. Always.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# TABLE 1: TIER_MAP — Issue Label → Tier (0-3)
# ═══════════════════════════════════════════════════════════════
# Tier 0 = Emergency (may need 911)     → Priority: Critical
# Tier 1 = Critical infrastructure      → Priority: High
# Tier 2 = Dangerous infrastructure     → Priority: Medium
# Tier 3 = Public nuisance / health     → Priority: Low

TIER_MAP: Dict[str, int] = {
    # Tier 0 — Emergency
    "fire_hazard": 0,
    "fire": 0,                    # Alias for fire_hazard
    "car_accident": 0,
    "downed_power_line": 0,
    
    # Tier 1 — Critical Infrastructure
    "flooding": 1,
    "open_manhole": 1,
    "broken_traffic_signal": 1,
    "fallen_tree": 1,
    
    # Tier 2 — Dangerous Infrastructure
    "pothole": 2,
    "road_damage": 2,
    "broken_streetlight": 2,
    "water_leakage": 2,
    "damaged_sidewalk": 2,
    "damaged_traffic_sign": 2,
    "clogged_drain": 2,
    
    # Tier 3 — Public Nuisance / Health
    "garbage": 3,
    "dead_animal": 3,
    "abandoned_vehicle": 3,
    "graffiti_vandalism": 3,
    "park_playground_damage": 3,
}

# Known issue labels (all valid labels)
KNOWN_ISSUE_LABELS = set(TIER_MAP.keys())

# ═══════════════════════════════════════════════════════════════
# TABLE 2: TIER → PRIORITY mapping
# ═══════════════════════════════════════════════════════════════
TIER_PRIORITY_MAP: Dict[int, str] = {
    0: "Critical",
    1: "High",
    2: "Medium",
    3: "Low",
}

# Priority ordinal for comparisons (never compare strings lexicographically)
PRIORITY_ORDINAL: Dict[str, int] = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
}

# ═══════════════════════════════════════════════════════════════
# TABLE 3: EMERGENCY ADVISORIES — Liability-reviewed text
# ═══════════════════════════════════════════════════════════════
# These are BYTE-EXACT. Do not modify without legal review.

EMERGENCY_ADVISORIES: Dict[str, str] = {
    "fire_hazard": (
        "Some elements in this image may indicate an active fire hazard. "
        "If you are currently at the scene and believe there is any risk to "
        "people, structures, or the environment, consider contacting emergency "
        "services (911) directly, as EAiSER reports are not monitored in real "
        "time. You may continue filing this report for municipal follow-up "
        "once the scene is safe."
    ),
    "fire": (  # Same advisory for fire alias
        "Some elements in this image may indicate an active fire hazard. "
        "If you are currently at the scene and believe there is any risk to "
        "people, structures, or the environment, consider contacting emergency "
        "services (911) directly, as EAiSER reports are not monitored in real "
        "time. You may continue filing this report for municipal follow-up "
        "once the scene is safe."
    ),
    "car_accident": (
        "Some elements in this image may indicate a recent vehicle collision. "
        "If anyone at the scene appears to need medical attention or if the "
        "roadway is unsafe, consider contacting emergency services (911) "
        "directly, as EAiSER reports are not monitored in real time. You may "
        "continue filing this report to document property or infrastructure "
        "damage."
    ),
    "downed_power_line": (
        "Some elements in this image may indicate a downed or damaged power "
        "line. Downed lines can remain energized even when they appear "
        "inactive. If you are near the scene, consider maintaining a safe "
        "distance and contacting your local utility or emergency services "
        "(911) directly, as EAiSER reports are not monitored in real time. "
        "You may continue filing this report once you are at a safe distance."
    ),
}

# Precedence when multiple Tier 0 issues coexist
# (Downed Power Line > Fire Hazard > Car Accident)
ADVISORY_PRECEDENCE = ["downed_power_line", "fire_hazard", "fire", "car_accident"]


# ═══════════════════════════════════════════════════════════════
# FUNCTION: Get tier for an issue type
# ═══════════════════════════════════════════════════════════════
def get_tier(issue_type: str) -> int:
    """Get the authoritative tier for an issue type. Returns -1 for unknown."""
    normalized = issue_type.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return TIER_MAP.get(normalized, -1)


def get_priority_from_tier(tier: int) -> str:
    """Get priority string from tier number."""
    return TIER_PRIORITY_MAP.get(tier, "Medium")


def get_priority_for_issue(issue_type: str) -> str:
    """Get the authoritative priority for an issue type."""
    tier = get_tier(issue_type)
    if tier == -1:
        return "Medium"  # Unknown issues default to Medium
    return get_priority_from_tier(tier)


def is_emergency(issue_type: str) -> bool:
    """Check if an issue type is Tier 0 (emergency)."""
    return get_tier(issue_type) == 0


def get_emergency_advisory(issue_type: str) -> Optional[str]:
    """Get the canonical emergency advisory text for a Tier 0 issue."""
    normalized = issue_type.lower().replace(" ", "_").replace("-", "_")
    return EMERGENCY_ADVISORIES.get(normalized)


# ═══════════════════════════════════════════════════════════════
# SCENARIO DETERMINATION (A-F)
# ═══════════════════════════════════════════════════════════════
def determine_scenario(
    known_issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]]
) -> Tuple[str, bool, bool]:
    """
    Determine the scenario (A-F) based on actual issue composition.
    
    Returns: (scenario_letter, internal_review_required, emergency_911)
    
    Decision tree (from V4 spec):
        Has Tier 0? → F (review=true, emergency=true)
        No → Has unknown?
            Yes → Has known?
                Yes → Multi-tier? → D (review=true) else B (review=true)
                No → E (review=true)
            No → Has known?
                Yes → Multi-tier? → C (review=false) else A (review=false)
                No → ERROR (treat as E)
    """
    has_tier_0 = any(get_tier(k.get("issue", "")) == 0 for k in known_issues)
    has_known = len(known_issues) > 0
    has_unknown = len(unknown_issues) > 0
    
    # Count distinct tiers among known issues
    known_tiers = set()
    for k in known_issues:
        t = get_tier(k.get("issue", ""))
        if t >= 0:
            known_tiers.add(t)
    multi_tier = len(known_tiers) >= 2
    
    if has_tier_0:
        return "F", True, True
    
    if has_unknown:
        if has_known:
            if multi_tier:
                return "D", True, False
            else:
                return "B", True, False
        else:
            return "E", True, False
    else:
        if has_known:
            if multi_tier:
                return "C", False, False
            else:
                return "A", False, False
        else:
            # No issues at all — treat as E (requires review)
            return "E", True, False


# ═══════════════════════════════════════════════════════════════
# RECONCILER — Enforce consistency on AI output
# ═══════════════════════════════════════════════════════════════
def reconcile_classification(
    issue_type: str,
    severity: str,
    confidence: float,
    priority: str,
    category: str = "public"
) -> Dict[str, Any]:
    """
    V4 Reconciler for classification output.
    Enforces tier-based priority and emergency handling.
    
    Returns a dict with corrected values + metadata about what was fixed.
    """
    soft_fixes = []
    
    # Normalize issue type
    normalized_type = issue_type.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    
    # 1. Tier lookup
    tier = get_tier(normalized_type)
    
    if tier >= 0:
        # Known issue — enforce tier-based priority
        correct_priority = get_priority_from_tier(tier)
        if priority != correct_priority:
            soft_fixes.append(f"Priority corrected: '{priority}' → '{correct_priority}' (Tier {tier})")
            priority = correct_priority
        
        # Enforce severity consistency with tier
        tier_severity_map = {0: "High", 1: "High", 2: "Medium", 3: "Low"}
        correct_severity = tier_severity_map.get(tier, severity)
        if severity.title() != correct_severity:
            soft_fixes.append(f"Severity corrected: '{severity}' → '{correct_severity}' (Tier {tier})")
            severity = correct_severity
    else:
        # Unknown issue — keep AI's assessment but clamp confidence
        if confidence > 80:
            soft_fixes.append(f"Unknown issue confidence clamped: {confidence} → 70")
            confidence = min(confidence, 70.0)
        # Unknown issues always need review
        priority = priority if priority in ["High", "Medium", "Low"] else "Medium"
    
    # 2. Emergency check
    emergency_911 = tier == 0
    emergency_advisory = None
    if emergency_911:
        emergency_advisory = get_emergency_advisory(normalized_type)
        if not emergency_advisory:
            # Fallback: use fire advisory for any Tier 0
            emergency_advisory = EMERGENCY_ADVISORIES.get("fire_hazard")
        soft_fixes.append(f"Emergency 911 flag set for Tier 0 issue: {normalized_type}")
    
    # 3. Scenario determination (for single issues)
    known_issues = [{"issue": normalized_type}] if tier >= 0 else []
    unknown_issues = [{"issue": normalized_type}] if tier < 0 else []
    scenario, review_required, _ = determine_scenario(known_issues, unknown_issues)
    
    # 4. Review requirement (from V4 spec: B, D, E, F require review)
    if review_required:
        soft_fixes.append(f"Review required for scenario {scenario}")
    
    # Low confidence always needs review
    if confidence < 70:
        review_required = True
        soft_fixes.append(f"Review required due to low confidence ({confidence}%)")
    
    return {
        "issue_type": normalized_type,
        "tier": tier,
        "severity": severity,
        "confidence": confidence,
        "priority": priority,
        "category": category,
        "emergency_911": emergency_911,
        "emergency_advisory": emergency_advisory,
        "scenario": scenario,
        "internal_review_required": review_required,
        "_soft_fixes": soft_fixes,
        "_tier_source": "TIER_MAP" if tier >= 0 else "unknown",
    }


def compute_final_priority(
    known_issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]]
) -> str:
    """
    Compute final priority from V4 spec:
    - Tier 0 present → Critical
    - Else max(known tier priority, max unknown severity)
    """
    max_ordinal = 0
    
    for k in known_issues:
        tier = get_tier(k.get("issue", ""))
        if tier == 0:
            return "Critical"
        priority = get_priority_from_tier(tier) if tier >= 0 else "Medium"
        max_ordinal = max(max_ordinal, PRIORITY_ORDINAL.get(priority, 2))
    
    for u in unknown_issues:
        sev = u.get("severity", "Medium")
        max_ordinal = max(max_ordinal, PRIORITY_ORDINAL.get(sev, 2))
    
    # Reverse lookup
    for p, o in PRIORITY_ORDINAL.items():
        if o == max_ordinal:
            return p
    
    return "Medium"


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Full reconcile for a report from classify_issue
# ═══════════════════════════════════════════════════════════════
def reconcile_and_enrich(
    issue_type: str,
    severity: str,
    confidence: float,
    category: str,
    priority: str
) -> Dict[str, Any]:
    """
    Main entry point. Call after classify_issue() to enforce V4 rules.
    Returns enriched data with tier, emergency info, and scenario.
    """
    result = reconcile_classification(issue_type, severity, confidence, priority, category)
    
    if result["_soft_fixes"]:
        logger.info(f"V4 Reconciler applied {len(result['_soft_fixes'])} fixes for '{issue_type}':")
        for fix in result["_soft_fixes"]:
            logger.info(f"  ↳ {fix}")
    
    return result
