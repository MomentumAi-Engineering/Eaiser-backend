"""
Post-Classification Engine
===========================
Contains two strict override layers applied AFTER AI classification:

1. Fire Detection Override Engine (Task 6)
   - Distinguishes controlled fires from dangerous fires
   - Overrides AI output when clear indicators are present

2. Confidence Control System (Task 7)
   - Clamps confidence between 0 and 100
   - Enforces consistency between severity and confidence
   - Adds low_confidence flag when confidence < 60
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# TASK 6: FIRE DETECTION OVERRIDE ENGINE
# =========================================================================

CONTROLLED_FIRE_INDICATORS = [
    "campfire", "bonfire", "bbq", "barbecue", "fire pit",
    "festival", "diya", "incense", "ceremony", "controlled burn",
]

DANGER_INDICATORS = [
    "spreading", "out of control", "wildfire", "structure burning",
    "house fire", "building fire", "explosion", "large smoke plume",
]


def apply_fire_detection_override(
    description: str,
    issue_detected: bool,
    issue_type: str,
    severity: str,
    ai_confidence_percent: float,
) -> Dict[str, Any]:
    """
    Strict override layer AFTER AI classification.
    Checks description for controlled fire vs. danger indicators.

    Returns a dict with potentially overridden values:
        issue_detected, issue_type, severity, ai_confidence_percent, fire_override_applied
    """
    desc_lower = description.lower() if description else ""

    has_controlled = any(indicator in desc_lower for indicator in CONTROLLED_FIRE_INDICATORS)
    has_danger = any(indicator in desc_lower for indicator in DANGER_INDICATORS)

    result = {
        "issue_detected": issue_detected,
        "issue_type": issue_type,
        "severity": severity,
        "ai_confidence_percent": ai_confidence_percent,
        "fire_override_applied": False,
    }

    if has_controlled and not has_danger:
        # Controlled fire: override to safe
        result["issue_detected"] = False
        result["issue_type"] = "other"
        result["severity"] = "Low"
        result["ai_confidence_percent"] = min(ai_confidence_percent, 35)
        result["fire_override_applied"] = True
        logger.info(
            f"🔥 Fire Override: Controlled fire detected. "
            f"Overriding issue_detected=False, severity=Low, confidence={result['ai_confidence_percent']}"
        )
    elif has_danger:
        # Danger: override to high
        result["issue_detected"] = True
        result["severity"] = "High" if severity not in ("High", "Critical") else severity
        result["ai_confidence_percent"] = max(ai_confidence_percent, 85)
        result["fire_override_applied"] = True
        logger.info(
            f"🔥 Fire Override: Danger indicators detected. "
            f"Overriding issue_detected=True, severity={result['severity']}, confidence={result['ai_confidence_percent']}"
        )

    return result


# =========================================================================
# TASK 7: CONFIDENCE CONTROL SYSTEM
# =========================================================================

def apply_confidence_controls(
    ai_confidence_percent: float,
    issue_type: str,
    severity: str,
    description: str = "",
) -> Dict[str, Any]:
    """
    Enforce hard confidence boundaries.

    Returns a dict with:
        ai_confidence_percent, severity, low_confidence (bool)
    """
    # 1. Clamp confidence between 0 and 100
    confidence = max(0.0, min(100.0, float(ai_confidence_percent)))

    # 2. If issue_type == "unknown", confidence <= 50
    if str(issue_type).lower() == "unknown":
        confidence = min(confidence, 50.0)

    # 5. Prevent confidence > 95 unless severity == "High" AND danger indicators present
    desc_lower = str(description).lower() if description else ""
    has_danger = any(indicator in desc_lower for indicator in DANGER_INDICATORS)

    if confidence > 95:
        if not (str(severity).lower() == "high" and has_danger):
            confidence = 95.0

    # 3. If severity == "High" but confidence < 60: downgrade severity to "Medium"
    current_severity = severity
    if str(severity).lower() == "high" and confidence < 60:
        current_severity = "Medium"
        logger.info(
            f"📊 Confidence Control: Downgraded severity from High to Medium "
            f"(confidence={confidence}%)"
        )

    # 4. If confidence < 60: add low_confidence flag
    low_confidence = confidence < 60

    if low_confidence:
        logger.info(f"📊 Confidence Control: low_confidence=True (confidence={confidence}%)")

    return {
        "ai_confidence_percent": confidence,
        "severity": current_severity,
        "low_confidence": low_confidence,
    }
