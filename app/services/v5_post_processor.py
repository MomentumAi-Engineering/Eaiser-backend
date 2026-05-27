"""
EAiSER V5 Post-Processor
==========================
Runs AFTER AI analysis to enforce safety rules deterministically.

Responsibilities:
  1. Primary issue selection (deterministic tiebreak)
  2. 5 auto-escalation rules (compound emergency promotion)
  3. Linked relationship upgrade (co_present → compound_emergency)
  4. Confidence-based routing bands
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

from services.v5_severity_engine import (
    normalize_label, is_tier_0, TIER_0_CATEGORIES,
    SEVERITY_ORDINAL, compute_severity_from_factors,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# AUTO-ESCALATION RULES (5 rules from V5 spec)
# ═══════════════════════════════════════════════════════════════

def apply_auto_escalation(
    issues: List[Dict[str, Any]],
    scene_text: str = "",
    visual_observations: Dict[str, List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Apply 5 auto-escalation rules to promote compound scenarios to Tier 0.

    Returns: (updated_issues, escalation_log)
    """
    escalation_log = []
    obs = visual_observations or {}
    all_obs_text = " ".join(
        o for img_obs in obs.values() if isinstance(img_obs, list) for o in img_obs
    ).lower()
    combined_text = (scene_text + " " + all_obs_text).lower()

    known_labels = {normalize_label(i.get("label", i.get("issue", ""))) for i in issues}

    # ── RULE 1: Standing water + life-threat cues → Active Flooding (Tier 0) ──
    flooding_labels = {"flooding", "standing_water", "flooding___standing_water"}
    life_threat_cues = [
        "submerged", "trapped", "rising water", "water rescue",
        "vehicles stuck", "road impassable", "chest-deep",
        "waist-deep", "knee-deep", "swift water", "flash flood",
    ]
    if known_labels & flooding_labels:
        if any(cue in combined_text for cue in life_threat_cues):
            if "active_flooding" not in known_labels:
                issues.append({
                    "label": "active_flooding",
                    "issue": "Active Flooding",
                    "confidence": 0.85,
                    "risk_factors": {"presence_of_hazard": True, "active_vs_latent": "active"},
                    "_escalation_source": "rule_1_flooding_life_threat",
                    "_inferred": True,
                })
                known_labels.add("active_flooding")
                escalation_log.append(
                    "Rule 1: Standing water + life-threat cues → Active Flooding (Tier 0)"
                )
                logger.warning("🚨 Auto-escalation Rule 1: Promoted to Active Flooding")

    # ── RULE 2: Fallen tree + utility pole/power line damage → Downed Power Line (Tier 0) ──
    tree_labels = {"fallen_tree", "hazardous_tree"}
    power_cues = [
        "power line", "utility pole", "electrical", "cable",
        "wire", "transformer", "sparking", "arcing", "energized",
        "snapped pole", "broken pole", "pole damage",
    ]
    if known_labels & tree_labels:
        if any(cue in combined_text for cue in power_cues):
            if "downed_power_line" not in known_labels:
                issues.append({
                    "label": "downed_power_line",
                    "issue": "Downed Power Line",
                    "confidence": 0.80,
                    "risk_factors": {"presence_of_hazard": True, "utility_service_impact": True},
                    "_escalation_source": "rule_2_tree_power_line",
                    "_inferred": True,
                })
                known_labels.add("downed_power_line")
                escalation_log.append(
                    "Rule 2: Fallen tree + power line damage → Downed Power Line (Tier 0)"
                )
                logger.warning("🚨 Auto-escalation Rule 2: Promoted to Downed Power Line")

    # ── RULE 2.5: Fallen tree on vehicle → Car Accident (Tier 0) ──
    # Strict vehicle keywords ONLY. Loose phrases like "fallen onto",
    # "crushed", "roof damage", "rests across" used to false-positive on
    # tree-on-house scenes (the roof here is the building's, not a car's).
    vehicle_cues = [
        "vehicle", "car ", " car,", " car.", "sedan", "truck", "suv",
        "automobile", "windshield", "parked car", "vehicle damage",
    ]
    if known_labels & tree_labels and any(cue in combined_text for cue in vehicle_cues):
        if "car_accident" not in known_labels:
            issues.append({
                "label": "car_accident",
                "issue": "Car Accident",
                "confidence": 0.80,
                "risk_factors": {"presence_of_hazard": True, "structural_risk": True},
                "_escalation_source": "rule_2_5_tree_on_vehicle",
                "_inferred": True,
            })
            known_labels.add("car_accident")
            escalation_log.append(
                "Rule 2.5: Fallen tree on vehicle → Car Accident (Tier 0)"
            )
            logger.warning("🚨 Auto-escalation Rule 2.5: Tree on vehicle → Car Accident")

    # ── RULE 3: Sidewalk/road damage + structural distress → Structural Collapse ──
    infra_labels = {"damaged_sidewalk", "road_damage", "pothole"}
    structural_cues = [
        "sinkhole", "subsidence", "cave-in", "collapse", "buckling",
        "foundation crack", "retaining wall", "bridge damage",
        "structural failure", "building lean",
    ]
    if known_labels & infra_labels:
        if any(cue in combined_text for cue in structural_cues):
            if "structural_collapse" not in known_labels:
                issues.append({
                    "label": "structural_collapse",
                    "issue": "Structural Collapse",
                    "confidence": 0.75,  # Capped at 0.80 per spec (we use 0.75 for inferred)
                    "risk_factors": {"structural_risk": True, "presence_of_hazard": True},
                    "_escalation_source": "rule_3_infra_structural_distress",
                    "_inferred": True,
                })
                known_labels.add("structural_collapse")
                escalation_log.append(
                    "Rule 3: Infrastructure + structural distress → Structural Collapse (conf capped 0.75)"
                )
                logger.warning("🚨 Auto-escalation Rule 3: Structural Collapse (conf capped)")

    # ── RULE 4: Water leak + gas infrastructure → High with cross_check ──
    water_labels = {"water_leakage", "water_leak"}
    gas_cues = ["gas line", "gas pipe", "gas main", "gas meter", "gas smell", "rotten egg"]
    if known_labels & water_labels:
        if any(cue in combined_text for cue in gas_cues):
            # Don't promote to Tier 0, but flag for cross-check
            for issue in issues:
                if normalize_label(issue.get("label", "")) in water_labels:
                    issue["_cross_check_required"] = True
                    issue["_cross_check_reason"] = "gas_infrastructure_proximity"
            escalation_log.append(
                "Rule 4: Water leak + gas infrastructure → cross_check_required flag"
            )
            logger.info("⚠️ Auto-escalation Rule 4: Water leak near gas → cross_check flagged")

    # ── RULE 5: Any issue + injured person → Person in Distress primary ──
    distress_cues = [
        "injured", "unconscious", "bleeding", "lying on ground",
        "medical emergency", "person down", "needs help",
        "hurt", "wounded", "collapsed person", "unresponsive",
    ]
    if any(cue in combined_text for cue in distress_cues):
        if "person_in_distress" not in known_labels:
            issues.insert(0, {  # Insert at position 0 = primary
                "label": "person_in_distress",
                "issue": "Person in Distress",
                "confidence": 0.85,
                "risk_factors": {"presence_of_hazard": True, "active_vs_latent": "active"},
                "_escalation_source": "rule_5_person_in_distress",
                "_inferred": True,
            })
            known_labels.add("person_in_distress")
            escalation_log.append(
                "Rule 5: Injured person detected → Person in Distress (Tier 0, primary)"
            )
            logger.warning("🚨 Auto-escalation Rule 5: Person in Distress promoted to primary")

    return issues, escalation_log


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE-BASED ROUTING BANDS
# ═══════════════════════════════════════════════════════════════

def get_confidence_band(confidence: float) -> Dict[str, Any]:
    """
    Determine routing behavior based on confidence.

    Returns: { band, show_911_banner, ops_review, user_message }
    """
    if confidence >= 0.85:
        return {
            "band": "high",
            "show_911_banner": True,  # If Tier 0
            "ops_review": False,
            "user_message": None,
        }
    elif confidence >= 0.70:
        return {
            "band": "medium_high",
            "show_911_banner": False,  # No 911 banner even if Tier 0
            "ops_review": True,
            "user_message": None,
        }
    elif confidence >= 0.50:
        return {
            "band": "medium",
            "show_911_banner": False,
            "ops_review": True,
            "user_message": "We're not fully sure about this classification. An operator will review.",
        }
    else:
        return {
            "band": "low",
            "show_911_banner": False,
            "ops_review": True,
            "user_message": "We couldn't confidently identify the issue. It will be reviewed manually.",
        }


# ═══════════════════════════════════════════════════════════════
# LINKED RELATIONSHIP UPGRADE
# ═══════════════════════════════════════════════════════════════

def upgrade_linked_relationships(issues: List[Dict[str, Any]]) -> List[str]:
    """
    Upgrade co_present → compound_emergency when both issues are High+.
    Returns list of upgrade log entries.
    """
    upgrade_log = []
    high_issues = []

    for issue in issues:
        label = normalize_label(issue.get("label", issue.get("issue", "")))
        severity = issue.get("computed_severity", "Low")
        if SEVERITY_ORDINAL.get(severity, 1) >= SEVERITY_ORDINAL["High"] or is_tier_0(label):
            high_issues.append(label)

    # If 2+ issues are High or above, mark relationships as compound_emergency
    if len(high_issues) >= 2:
        for issue in issues:
            if issue.get("_relationship") == "co_present":
                issue["_relationship"] = "compound_emergency"
                upgrade_log.append(
                    f"Upgraded '{issue.get('label', '')}' from co_present → compound_emergency"
                )

    return upgrade_log


# ═══════════════════════════════════════════════════════════════
# MAIN POST-PROCESSOR
# ═══════════════════════════════════════════════════════════════

def run_post_processor(
    issues: List[Dict[str, Any]],
    unknown_issues: List[Dict[str, Any]],
    scene_text: str = "",
    visual_observations: Dict[str, List[str]] = None,
) -> Dict[str, Any]:
    """
    Main post-processing pipeline. Runs after AI analysis.

    Returns:
        {
            "issues": [...],
            "unknown_issues": [...],
            "escalation_log": [...],
            "confidence_band": {...},
            "post_processing": {
                "auto_escalation_applied": bool,
                "escalation_count": int,
                "relationship_upgrades": [...],
                "primary_confidence": float,
            }
        }
    """
    # Step 1: Auto-escalation
    issues, escalation_log = apply_auto_escalation(
        issues, scene_text, visual_observations
    )

    # Step 2: Compute severity for each issue
    for issue in issues:
        label = issue.get("label", issue.get("issue", ""))
        rf = issue.get("risk_factors", {})
        severity, is_emg, expl = compute_severity_from_factors(label, rf)
        issue["computed_severity"] = severity
        issue["is_emergency"] = is_emg
        issue["severity_explanations"] = expl

    # Step 3: Linked relationship upgrade
    upgrade_log = upgrade_linked_relationships(issues)

    # Step 4: Determine primary issue confidence band
    primary_conf = 0.75  # Default
    if issues:
        primary_conf = issues[0].get("confidence", 0.75)
        if isinstance(primary_conf, str):
            primary_conf = {"High": 0.95, "Medium": 0.75, "Low": 0.45}.get(primary_conf, 0.75)
    conf_band = get_confidence_band(primary_conf)

    # Override 911 banner based on confidence band
    if not conf_band["show_911_banner"]:
        for issue in issues:
            if is_tier_0(issue.get("label", "")):
                issue["_911_banner_suppressed"] = True
                issue["_911_suppression_reason"] = f"confidence={primary_conf:.2f} below 0.85"

    if escalation_log:
        logger.info(f"🔧 V5 Post-Processor: {len(escalation_log)} escalations applied")
        for entry in escalation_log:
            logger.info(f"  ↳ {entry}")

    return {
        "issues": issues,
        "unknown_issues": unknown_issues,
        "escalation_log": escalation_log,
        "confidence_band": conf_band,
        "post_processing": {
            "auto_escalation_applied": len(escalation_log) > 0,
            "escalation_count": len(escalation_log),
            "relationship_upgrades": upgrade_log,
            "primary_confidence": primary_conf,
        }
    }
