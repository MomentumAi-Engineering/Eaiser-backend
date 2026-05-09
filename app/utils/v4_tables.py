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
    "vehicle_damage": 0,          # Compound: tree/debris damaged vehicle
    "tree_on_vehicle": 0,         # Compound: tree fell on car
    "tree_on_car": 0,             # Alias for tree_on_vehicle
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


# ═══════════════════════════════════════════════════════════════
# SIMI LEVEL 3: Multi-Issue Reconciliation
# ═══════════════════════════════════════════════════════════════
# When a single image contains MULTIPLE issues (e.g., car accident +
# pothole + garbage), this function reconciles ALL of them and
# determines the overall priority from the highest-tier issue.

def reconcile_multi_issue(v3_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    V4 Reconciler for FULL SIMI output (all issues from a single image).
    
    Enforces:
    - Correct tiers on every known issue (from TIER_MAP)
    - Overall priority = highest-tier issue's priority
    - Emergency flags if ANY issue is Tier 0
    - Correct scenario (A-F) based on full issue set
    - Deterministic ordered_issue_list
    
    Args:
        v3_data: The raw V3 report data with known_issues[], unknown_issues[], etc.
    
    Returns:
        Dict with reconciled data + metadata:
        {
            "known_issues": [...],        # Tier-corrected
            "unknown_issues": [...],
            "ordered_issue_list": [...],   # Re-sorted deterministically
            "primary_issue": str,          # Highest priority issue label
            "primary_tier": int,           # Tier of primary issue
            "scenario": str,              # A-F
            "final_priority": str,        # Critical/High/Medium/Low
            "final_severity": str,        # Based on highest tier
            "emergency_911": bool,
            "emergency_advisory": str|None,
            "internal_review_required": bool,
            "total_issues": int,
            "issue_summary": str,         # "Car Accident, Pothole, Garbage"
            "_soft_fixes": [...]
        }
    """
    soft_fixes = []
    known = list(v3_data.get("known_issues", []))
    unknown = list(v3_data.get("unknown_issues", []))
    
    # ── Step 1: Fix tiers on all known issues ──
    for k in known:
        issue_label = k.get("issue", "")
        # Normalize for TIER_MAP lookup
        normalized = issue_label.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        correct_tier = TIER_MAP.get(normalized, -1)
        
        if correct_tier >= 0:
            if k.get("tier") != correct_tier:
                soft_fixes.append(f"Tier corrected for '{issue_label}': {k.get('tier')} → {correct_tier}")
                k["tier"] = correct_tier
        else:
            # Label not in TIER_MAP — move to unknown if it snuck in
            soft_fixes.append(f"Unknown label '{issue_label}' in known_issues — tier set to -1")
            k["tier"] = -1
    
    # ── Step 1.5: Consequential Issue Inference ──
    # When the AI detects one issue but the scene clearly shows compound damage,
    # auto-inject the missing consequential issues so ALL departments are notified.
    # This is deterministic — based on scene text + known issue combinations.
    known_labels = {k.get("issue", "").lower().replace(" ", "_").replace("-", "_") for k in known}
    scene_text = (
        v3_data.get("scene_description", "") + " " +
        " ".join(
            obs
            for img_obs in v3_data.get("visual_observations", {}).values()
            if isinstance(img_obs, list)
            for obs in img_obs
        )
    ).lower()
    
    # RULE 1: Fallen tree + vehicle damage context → inject car_accident (Tier 0)
    # Ensures Police + Emergency are notified alongside Public Works
    vehicle_damage_cues = [
        "sedan", "vehicle", "car ", "automobile", "truck", "suv",
        "roof damage", "body damage", "windshield", "crushed",
        "vehicle's roof", "vehicle damage", "damage to the vehicle",
        "fallen onto a", "fallen on a", "rests across the vehicle",
        "resting on", "onto a parked", "penetrated the"
    ]
    if ("fallen_tree" in known_labels or "fallen___hazardous_tree" in known_labels) \
            and "car_accident" not in known_labels:
        if any(cue in scene_text for cue in vehicle_damage_cues):
            known.append({
                "issue": "car_accident",
                "tier": 0,
                "confidence": "High",
                "image_source": ["image_1"],
                "cross_image_confirmed": False,
                "_inferred": True
            })
            known_labels.add("car_accident")
            soft_fixes.append("Inferred 'car_accident' from fallen_tree + vehicle damage in scene description")
            logger.info("🚗 Consequential inference: car_accident added (fallen tree + vehicle damage detected)")
    
    # RULE 2: Downed power line + fire context → inject fire_hazard
    fire_cues = ["spark", "fire", "flame", "smoke", "burning", "smoldering", "charred"]
    if ("downed_power_line" in known_labels) and "fire_hazard" not in known_labels and "fire" not in known_labels:
        if any(cue in scene_text for cue in fire_cues):
            known.append({
                "issue": "fire_hazard",
                "tier": 0,
                "confidence": "Medium",
                "image_source": ["image_1"],
                "cross_image_confirmed": False,
                "_inferred": True
            })
            known_labels.add("fire_hazard")
            soft_fixes.append("Inferred 'fire_hazard' from downed_power_line + fire cues in scene")
    
    # RULE 3: Car accident + road debris → inject road_damage
    road_debris_cues = ["debris", "scattered", "obstructed", "blocked", "wreckage on road", "road surface"]
    if ("car_accident" in known_labels) and "road_damage" not in known_labels:
        if any(cue in scene_text for cue in road_debris_cues):
            known.append({
                "issue": "road_damage",
                "tier": 2,
                "confidence": "Medium",
                "image_source": ["image_1"],
                "cross_image_confirmed": False,
                "_inferred": True
            })
            known_labels.add("road_damage")
            soft_fixes.append("Inferred 'road_damage' from car_accident + road debris cues")
    
    # ── Step 2: Dedup known issues by label ──
    seen_known = {}
    for k in known:
        label = k.get("issue", "").strip()
        if label in seen_known:
            # Keep higher confidence
            existing_conf = {"High": 3, "Medium": 2, "Low": 1}.get(seen_known[label].get("confidence", "Low"), 1)
            new_conf = {"High": 3, "Medium": 2, "Low": 1}.get(k.get("confidence", "Low"), 1)
            if new_conf > existing_conf:
                seen_known[label] = k
            soft_fixes.append(f"Dedup: duplicate known issue '{label}' merged")
        else:
            seen_known[label] = k
    known = list(seen_known.values())
    
    # ── Step 3: Dedup unknown issues by normalized label ──
    seen_unknown = {}
    for u in unknown:
        label_norm = u.get("issue", "").strip().lower().replace("  ", " ")
        if label_norm in seen_unknown:
            existing_sev = {"High": 3, "Medium": 2, "Low": 1}.get(seen_unknown[label_norm].get("severity", "Low"), 1)
            new_sev = {"High": 3, "Medium": 2, "Low": 1}.get(u.get("severity", "Low"), 1)
            if new_sev > existing_sev:
                seen_unknown[label_norm] = u
            soft_fixes.append(f"Dedup: duplicate unknown issue '{u.get('issue')}' merged")
        else:
            seen_unknown[label_norm] = u
    unknown = list(seen_unknown.values())
    
    # ── Step 4: Determine scenario from FULL issue set ──
    scenario, review_required, emergency = determine_scenario(known, unknown)
    
    # ── Step 5: Compute final priority from ALL issues ──
    final_priority = compute_final_priority(known, unknown)
    
    # ── Step 6: Emergency advisory ──
    emergency_advisory = None
    if emergency:
        # Pick advisory by precedence: Downed Power Line > Fire Hazard > Car Accident
        for label in ADVISORY_PRECEDENCE:
            for k in known:
                norm_label = k.get("issue", "").lower().replace(" ", "_").replace("-", "_").replace("/", "_")
                if norm_label == label:
                    emergency_advisory = EMERGENCY_ADVISORIES.get(label)
                    break
            if emergency_advisory:
                break
        if not emergency_advisory:
            emergency_advisory = EMERGENCY_ADVISORIES.get("fire_hazard")
        soft_fixes.append("Emergency 911 flag set due to Tier 0 issue")
    
    # ── Step 7: Build deterministic ordered_issue_list ──
    all_items = []
    for k in known:
        tier = k.get("tier", 3)
        conf_ord = {"High": 3, "Medium": 2, "Low": 1}.get(k.get("confidence", "Low"), 1)
        all_items.append({
            "issue": k.get("issue", ""),
            "type": "Known",
            "tier_or_severity": f"Tier {tier}",
            "_sort_tier": tier,
            "_sort_signal": conf_ord,
            "_sort_label": k.get("issue", "").lower(),
        })
    for u in unknown:
        sev = u.get("severity", "Low")
        sev_ord = {"High": 3, "Medium": 2, "Low": 1}.get(sev, 1)
        all_items.append({
            "issue": u.get("issue", ""),
            "type": "Unknown",
            "tier_or_severity": sev,
            "_sort_tier": 4,  # Unknown after all tiers
            "_sort_signal": sev_ord,
            "_sort_label": u.get("issue", "").lower(),
        })
    
    # Sort: tier asc → signal desc → label asc
    all_items.sort(key=lambda x: (x["_sort_tier"], -x["_sort_signal"], x["_sort_label"]))
    
    ordered_issue_list = []
    for idx, item in enumerate(all_items):
        ordered_issue_list.append({
            "rank": idx + 1,
            "issue": item["issue"],
            "type": item["type"],
            "tier_or_severity": item["tier_or_severity"],
        })
    
    # ── Step 8: Determine primary issue (rank 1) ──
    primary_issue = ordered_issue_list[0]["issue"] if ordered_issue_list else "unknown"
    primary_tier = all_items[0]["_sort_tier"] if all_items else -1
    
    # ── Step 9: Map priority → severity for backward compat ──
    severity_map = {"Critical": "High", "High": "High", "Medium": "Medium", "Low": "Low"}
    final_severity = severity_map.get(final_priority, "Medium")
    
    # ── Step 10: Build human-readable issue summary ──
    issue_labels = [item["issue"] for item in ordered_issue_list]
    issue_summary = ", ".join(issue_labels) if issue_labels else "Unknown"
    
    total_issues = len(ordered_issue_list)
    
    if soft_fixes:
        logger.info(f"V4 SIMI Reconciler applied {len(soft_fixes)} fixes across {total_issues} issues:")
        for fix in soft_fixes:
            logger.info(f"  ↳ {fix}")
    
    logger.info(
        f"🔍 SIMI Result: {total_issues} issues detected — "
        f"Primary='{primary_issue}' (Tier {primary_tier}), "
        f"Priority={final_priority}, Scenario={scenario}, "
        f"Emergency={emergency}"
    )
    
    return {
        "known_issues": known,
        "unknown_issues": unknown,
        "ordered_issue_list": ordered_issue_list,
        "primary_issue": primary_issue,
        "primary_tier": primary_tier,
        "scenario": scenario,
        "final_priority": final_priority,
        "final_severity": final_severity,
        "emergency_911": emergency,
        "emergency_advisory": emergency_advisory,
        "internal_review_required": review_required,
        "total_issues": total_issues,
        "issue_summary": issue_summary,
        "_soft_fixes": soft_fixes,
    }


def get_departments_for_all_issues(
    known_issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]]
) -> List[str]:
    """
    Resolve ALL departments that should receive the report based on
    every detected issue — not just the primary one.
    
    Returns a deduplicated list of department keys (e.g., ["emergency", "public_works", "sanitation"]).
    """
    # Issue type → department mapping (mirrors issue_department_map.json)
    # This is a fallback; the actual JSON file is loaded by authority_service.py
    ISSUE_DEPT_FALLBACK = {
        "fire_hazard": ["fire", "emergency"],
        "fire": ["fire", "emergency"],
        "car_accident": ["police", "emergency"],
        "downed_power_line": ["electric_utility", "emergency", "fire"],
        "flooding": ["public_works", "emergency"],
        "open_manhole": ["public_works"],
        "broken_traffic_signal": ["transportation"],
        "fallen_tree": ["public_works", "parks"],
        "pothole": ["public_works", "transportation"],
        "road_damage": ["public_works", "transportation"],
        "broken_streetlight": ["electric_utility", "public_works"],
        "water_leakage": ["water_utility", "public_works"],
        "damaged_sidewalk": ["public_works"],
        "damaged_traffic_sign": ["transportation"],
        "clogged_drain": ["public_works", "water_utility"],
        "garbage": ["sanitation"],
        "dead_animal": ["animal_control", "sanitation"],
        "abandoned_vehicle": ["police", "parking"],
        "graffiti_vandalism": ["police", "public_works"],
        "park_playground_damage": ["parks"],
    }
    
    all_depts = set()
    
    for k in known_issues:
        issue_label = k.get("issue", "")
        normalized = issue_label.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        depts = ISSUE_DEPT_FALLBACK.get(normalized, ["general"])
        all_depts.update(depts)
    
    for u in unknown_issues:
        # Unknown issues always go to general for review
        all_depts.add("general")
    
    return list(all_depts)
