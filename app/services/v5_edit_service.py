"""
EAiSER V5 Edit Service
========================
Handles resident edits to reports, with Tier 0 locking.

Rules:
  - Non-emergency issues: resident can edit severity, description, label
  - Tier 0 (Emergency): LOCKED — resident can only submit a dispute note
  - Every edit is logged in edit_log[] for auditability
  - Editing marks summary_freshness as "stale_pending_refresh"
  - Suppression detection: if resident downgrades all issues to Low, flag for ops review
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from services.v5_severity_engine import (
    normalize_label, is_tier_0, SEVERITY_ORDINAL, SEVERITY_LEVELS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# EDIT VALIDATORS
# ═══════════════════════════════════════════════════════════════

def can_edit_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if a specific issue is editable by the resident.

    Returns: { editable: bool, reason: str, editable_fields: list }
    """
    label = normalize_label(issue.get("label", issue.get("issue", "")))

    if is_tier_0(label):
        return {
            "editable": False,
            "reason": f"Tier 0 emergency '{label}' cannot be edited. Submit a dispute note instead.",
            "editable_fields": [],
            "can_dispute": True,
        }

    return {
        "editable": True,
        "reason": "Non-emergency issue — editable by resident.",
        "editable_fields": ["severity", "description", "label"],
        "can_dispute": False,
    }


def validate_severity_edit(
    current_severity: str,
    new_severity: str,
    issue_label: str,
) -> Dict[str, Any]:
    """
    Validate a severity edit. Downgrades require confirmation.

    Returns: { valid: bool, requires_confirmation: bool, message: str }
    """
    if is_tier_0(issue_label):
        return {
            "valid": False,
            "requires_confirmation": False,
            "message": "Cannot change severity of emergency issues.",
        }

    if new_severity not in SEVERITY_LEVELS[:3]:  # Only Low, Medium, High for residents
        return {
            "valid": False,
            "requires_confirmation": False,
            "message": f"Invalid severity '{new_severity}'. Allowed: Low, Medium, High.",
        }

    current_ord = SEVERITY_ORDINAL.get(current_severity, 2)
    new_ord = SEVERITY_ORDINAL.get(new_severity, 2)

    if new_ord < current_ord:
        return {
            "valid": True,
            "requires_confirmation": True,
            "message": f"Downgrading from {current_severity} to {new_severity} — are you sure?",
        }

    return {
        "valid": True,
        "requires_confirmation": False,
        "message": "Severity update accepted.",
    }


# ═══════════════════════════════════════════════════════════════
# EDIT APPLICATION
# ═══════════════════════════════════════════════════════════════

def apply_edit(
    report: Dict[str, Any],
    issue_id: str,
    edits: Dict[str, Any],
    user_id: str,
) -> Dict[str, Any]:
    """
    Apply resident edits to a specific issue in a report.

    Args:
        report: The full V5 report
        issue_id: The r_{hash}_i{NN} ID of the issue to edit
        edits: { severity?: str, description?: str, label?: str }
        user_id: The editing user's ID

    Returns: { success: bool, report: dict, edit_entry: dict, warnings: list }
    """
    warnings = []

    # Find the issue
    target = None
    for issue in report.get("issues", []):
        if issue.get("issue_id") == issue_id:
            target = issue
            break

    if not target:
        return {
            "success": False,
            "report": report,
            "edit_entry": None,
            "warnings": [f"Issue {issue_id} not found in report."],
        }

    # Check editability
    edit_check = can_edit_issue(target)
    if not edit_check["editable"]:
        return {
            "success": False,
            "report": report,
            "edit_entry": None,
            "warnings": [edit_check["reason"]],
        }

    # Build edit log entry
    before = {
        "severity": target.get("computed_severity"),
        "description": target.get("description"),
        "label": target.get("label"),
    }

    # Apply edits
    if "severity" in edits:
        validation = validate_severity_edit(
            target.get("computed_severity", "Medium"),
            edits["severity"],
            target.get("label", ""),
        )
        if validation["valid"]:
            target["computed_severity"] = edits["severity"]
            target["_severity_source"] = "resident_edit"
        else:
            warnings.append(validation["message"])

    if "description" in edits:
        target["description"] = edits["description"]

    if "label" in edits:
        new_label = normalize_label(edits["label"])
        if not is_tier_0(new_label):
            target["label"] = edits["label"]
            target["issue"] = edits["label"]
        else:
            warnings.append("Cannot change label to a Tier 0 emergency category.")

    after = {
        "severity": target.get("computed_severity"),
        "description": target.get("description"),
        "label": target.get("label"),
    }

    # Create edit log entry
    edit_entry = {
        "edit_id": f"edit_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "issue_id": issue_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "before": before,
        "after": after,
        "edit_type": "resident_edit",
    }

    # Append to edit log
    if "edit_log" not in report:
        report["edit_log"] = []
    report["edit_log"].append(edit_entry)

    # Mark summary as stale
    report["summary_freshness"] = "stale_pending_refresh"
    report["report_status"] = "user_updated"

    # Suppression detection: if ALL issues are now Low, flag
    all_low = all(
        i.get("computed_severity", "Medium") == "Low"
        for i in report.get("issues", [])
    )
    if all_low and len(report.get("issues", [])) > 0:
        report["_suppression_detected"] = True
        report["requires_post_action_review"] = True
        warnings.append("All issues downgraded to Low — flagged for ops review.")
        logger.warning(f"🚩 Suppression detected: All issues in report set to Low by user {user_id}")

    return {
        "success": True,
        "report": report,
        "edit_entry": edit_entry,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════
# DISPUTE NOTES (Tier 0 only)
# ═══════════════════════════════════════════════════════════════

def submit_dispute(
    report: Dict[str, Any],
    issue_id: str,
    dispute_note: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Submit a dispute note for a Tier 0 (locked) issue.
    The classification does NOT change — only a note is logged for ops review.

    Returns: { success: bool, report: dict, dispute_entry: dict }
    """
    # Find the issue
    target = None
    for issue in report.get("issues", []):
        if issue.get("issue_id") == issue_id:
            target = issue
            break

    if not target:
        return {
            "success": False,
            "report": report,
            "dispute_entry": None,
        }

    # Only allow disputes on Tier 0
    if not is_tier_0(target.get("label", "")):
        return {
            "success": False,
            "report": report,
            "dispute_entry": None,
        }

    dispute_entry = {
        "edit_id": f"dispute_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "issue_id": issue_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "edit_type": "tier_0_dispute",
        "dispute_note": dispute_note[:500],  # Cap at 500 chars
        "classification_unchanged": True,
    }

    if "edit_log" not in report:
        report["edit_log"] = []
    report["edit_log"].append(dispute_entry)

    # Flag for ops review
    report["requires_post_action_review"] = True

    logger.info(f"📝 Tier 0 dispute submitted for {issue_id} by {user_id}")

    return {
        "success": True,
        "report": report,
        "dispute_entry": dispute_entry,
    }
