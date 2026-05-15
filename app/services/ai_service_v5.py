"""
EAiSER AI Service V5
======================
SIMI/MIMI router and V5 analysis pipeline.

Architecture:
  - 1 photo  → SIMI V5 (single-image analysis)
  - 2+ photos → MIMI V5 (multi-image clustering)
  - All outputs pass through post-processor and banned content filter
  - Returns V5 envelope with deterministic severity
"""

import json
import os
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.v5_severity_engine import (
    normalize_label, is_tier_0, is_known_label, generate_report_hash,
    generate_issue_id, compute_severity_from_factors, build_v5_report_envelope,
    DEFAULT_RISK_FACTORS, SEVERITY_ORDINAL, LABEL_ALIASES,
)
from services.v5_post_processor import run_post_processor
from services.v5_banned_content import filter_report

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# PROMPT LOADING
# ═══════════════════════════════════════════════════════════════

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_simi_v5_prompt() -> str:
    """Load the SIMI V5 system prompt."""
    path = os.path.join(PROMPTS_DIR, "simi_v5_prompt.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    logger.error("❌ SIMI V5 prompt file not found!")
    return ""


def load_mimi_v5_prompt() -> str:
    """Load the MIMI V5 system prompt (multi-image)."""
    path = os.path.join(PROMPTS_DIR, "mimi_v5_prompt.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    # Fallback: use SIMI prompt with multi-image preamble
    logger.warning("⚠️ MIMI V5 prompt not found, using SIMI with multi-image wrapper")
    simi = load_simi_v5_prompt()
    return (
        "You are processing MULTIPLE images of the same civic scene. "
        "All images are from the same location and should be analyzed together. "
        "Cross-reference observations across images for consistency. "
        "If different images show the same issue, merge them into one issue entry. "
        "If images show different issues, list each separately.\n\n"
        + simi
    )


# ═══════════════════════════════════════════════════════════════
# V5 JSON PARSER
# ═══════════════════════════════════════════════════════════════

def parse_v5_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse AI output into V5 JSON. Handles markdown fencing and common issues.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.error(f"❌ Failed to parse V5 JSON output: {text[:200]}")
    return None


def normalize_v5_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize AI output to ensure all V5 fields exist and are correct types.
    """
    # Ensure top-level fields
    raw.setdefault("schema_version", "v5")
    raw.setdefault("simi_version", "v5.0.0")
    raw.setdefault("analysis_status", "issues_found")
    raw.setdefault("final_priority", "Low")
    raw.setdefault("primary_issue_id", None)
    raw.setdefault("truncated", False)
    raw.setdefault("truncated_count", 0)
    raw.setdefault("report_summary", "")
    raw.setdefault("issues", [])
    raw.setdefault("unknown_issues", [])
    raw.setdefault("edit_log", [])

    # Normalize each issue
    for issue in raw.get("issues", []):
        # Ensure label field exists (AI uses issue_type, we also use label)
        if "label" not in issue and "issue_type" in issue:
            issue["label"] = normalize_label(
                issue["issue_type"].lower().replace(" ", "_").replace("/", "_")
            )
        elif "label" in issue:
            issue["label"] = normalize_label(issue["label"])

        # Ensure confidence is float 0.0-1.0
        conf = issue.get("confidence", 0.75)
        if isinstance(conf, str):
            conf = {"High": 0.90, "Medium": 0.75, "Low": 0.50}.get(conf, 0.75)
        if conf > 1.0:
            conf = conf / 100.0  # Convert percentage to decimal
        issue["confidence"] = round(min(max(conf, 0.0), 1.0), 2)

        # Ensure risk_factors exist
        issue.setdefault("risk_factors", DEFAULT_RISK_FACTORS.copy())

        # Ensure is_tier_0
        issue["is_tier_0"] = is_tier_0(issue.get("label", ""))

        # Ensure other fields
        issue.setdefault("description", "")
        issue.setdefault("linked_to", [])
        issue.setdefault("linked_relationship", None)
        issue.setdefault("tier_0_advisory", None)
        issue.setdefault("escalation_source", None)
        issue.setdefault("hard_rule_triggered", None)

    # Normalize unknown issues
    for unknown in raw.get("unknown_issues", []):
        unknown.setdefault("confidence", 0.50)
        unknown.setdefault("severity", "Medium")
        unknown.setdefault("description", "")
        unknown.setdefault("suggested_label", "unknown_civic_issue")

    return raw


# ═══════════════════════════════════════════════════════════════
# V3 → V5 BRIDGE (backward compatibility)
# ═══════════════════════════════════════════════════════════════

def map_v3_to_v5(v3_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a V3/V4 analysis result to V5 format.
    Used during migration for reports analyzed by the old pipeline.
    """
    v5_issues = []

    for k in v3_data.get("known_issues", []):
        label = normalize_label(k.get("issue", ""))
        conf = k.get("confidence", "Medium")
        if isinstance(conf, str):
            conf = {"High": 0.90, "Medium": 0.75, "Low": 0.50}.get(conf, 0.75)

        v5_issues.append({
            "label": label,
            "issue_type": k.get("issue", label),
            "is_tier_0": is_tier_0(label),
            "severity": k.get("severity", "Medium"),
            "confidence": conf,
            "description": k.get("description", ""),
            "risk_factors": DEFAULT_RISK_FACTORS.copy(),
            "hard_rule_triggered": None,
            "linked_to": [],
            "linked_relationship": None,
            "tier_0_advisory": None,
            "escalation_source": None,
            "_migrated_from": "v3",
        })

    v5_unknown = []
    for u in v3_data.get("unknown_issues", []):
        v5_unknown.append({
            "issue": u.get("issue", "Unknown"),
            "suggested_label": u.get("issue", "unknown"),
            "severity": u.get("severity", "Medium"),
            "confidence": 0.50,
            "description": u.get("description", ""),
            "_migrated_from": "v3",
        })

    return {
        "schema_version": "v5",
        "simi_version": "v5.0.0-migrated",
        "analysis_status": "issues_found" if v5_issues else "unknown_only" if v5_unknown else "no_issue_detected",
        "final_priority": v3_data.get("priority", "Medium"),
        "primary_issue_id": None,
        "truncated": False,
        "truncated_count": 0,
        "report_summary": v3_data.get("scene_description", ""),
        "issues": v5_issues,
        "unknown_issues": v5_unknown,
        "edit_log": [],
    }


def map_v5_to_legacy(v5_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert V5 output back to V3/V4 format for backward compatibility.
    Used for clients that haven't upgraded to V5 yet.
    """
    known_issues = []
    for issue in v5_data.get("issues", []):
        known_issues.append({
            "issue": issue.get("issue_type", issue.get("label", "")),
            "tier": 0 if issue.get("is_tier_0") else 2,
            "confidence": "High" if issue.get("confidence", 0) >= 0.85 else "Medium" if issue.get("confidence", 0) >= 0.60 else "Low",
            "severity": issue.get("computed_severity", issue.get("severity", "Medium")),
            "description": issue.get("description", ""),
            "image_source": ["image_1"],
            "cross_image_confirmed": False,
        })

    unknown_issues = []
    for u in v5_data.get("unknown_issues", []):
        unknown_issues.append({
            "issue": u.get("suggested_label", u.get("issue", "Unknown")),
            "severity": u.get("severity", "Medium"),
            "description": u.get("description", ""),
        })

    # Build ordered issue list (V3 format)
    ordered = []
    for idx, issue in enumerate(known_issues):
        ordered.append({
            "rank": idx + 1,
            "issue": issue["issue"],
            "type": "Known",
            "tier_or_severity": f"Tier {issue['tier']}",
        })
    for idx, u in enumerate(unknown_issues):
        ordered.append({
            "rank": len(known_issues) + idx + 1,
            "issue": u["issue"],
            "type": "Unknown",
            "tier_or_severity": u["severity"],
        })

    emergency = v5_data.get("emergency", {})

    return {
        "schema_version": "v3-compat",
        "report_meta": {
            "report_type": "v5_migrated",
            "image_count": 1,
        },
        "scene_description": v5_data.get("report_summary", ""),
        "known_issues": known_issues,
        "unknown_issues": unknown_issues,
        "ordered_issue_list": ordered,
        "primary_issue": known_issues[0]["issue"] if known_issues else "unknown",
        "final_priority": v5_data.get("final_priority", "Medium"),
        "final_severity": v5_data.get("final_priority", "Medium"),
        "emergency_911": emergency.get("has_emergency", False),
        "emergency_advisory": emergency.get("banner"),
        "total_issues": len(known_issues) + len(unknown_issues),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════

async def analyze_image_v5(
    image_data: bytes,
    caption: str = "",
    generate_fn=None,
    exif_sidecar: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    V5 single-image analysis pipeline (SIMI path).

    Args:
        image_data: Raw image bytes
        caption: Optional resident caption
        generate_fn: async function(system_prompt, user_parts) -> str
        exif_sidecar: Optional EXIF metadata

    Returns: Complete V5 report envelope
    """
    report_hash = generate_report_hash(image_data)

    # Load system prompt
    system_prompt = load_simi_v5_prompt()

    # Build user message parts
    user_parts = []
    if caption:
        user_parts.append(
            f'RESIDENT CAPTION (optional, may be missing or unreliable):\n"{caption}"'
        )
    if exif_sidecar:
        exif_text = "EXIF SIDECAR (may be missing or partial):\n"
        for k, v in exif_sidecar.items():
            exif_text += f"- {k}: {v}\n"
        user_parts.append(exif_text)

    # Call AI
    if generate_fn:
        try:
            raw_text = await generate_fn(system_prompt, image_data, user_parts)
        except Exception as e:
            logger.error(f"❌ V5 AI call failed: {e}")
            return build_v5_report_envelope([], [], report_hash, f"Analysis error: {e}")
    else:
        logger.error("❌ No generate_fn provided for V5 analysis")
        return build_v5_report_envelope([], [], report_hash, "No AI engine available")

    # Parse AI output
    parsed = parse_v5_json(raw_text)
    if not parsed:
        return build_v5_report_envelope(
            [], [], report_hash, "Failed to parse AI output", image_count=1
        )

    # Normalize
    normalized = normalize_v5_output(parsed)

    # Run post-processor (auto-escalation, confidence bands)
    scene_text = normalized.get("report_summary", "") + " " + caption
    post_result = run_post_processor(
        issues=normalized.get("issues", []),
        unknown_issues=normalized.get("unknown_issues", []),
        scene_text=scene_text,
    )

    # Build V5 envelope with severity computation
    envelope = build_v5_report_envelope(
        issues=post_result["issues"],
        unknown_issues=post_result["unknown_issues"],
        report_hash=report_hash,
        scene_description=normalized.get("report_summary", ""),
        image_count=1,
    )

    # Attach post-processing metadata
    envelope["_post_processing"] = post_result["post_processing"]
    envelope["_post_processing"]["escalation_log"] = post_result["escalation_log"]
    envelope["_post_processing"]["confidence_band"] = post_result["confidence_band"]

    # Apply banned content filter
    envelope = filter_report(envelope)

    # Generate legacy-compatible output as well
    envelope["_legacy_v3"] = map_v5_to_legacy(envelope)

    logger.info(
        f"✅ V5 Analysis complete: {len(envelope['issues'])} issues, "
        f"priority={envelope['final_priority']}, "
        f"status={envelope['analysis_status']}"
    )

    return envelope


async def analyze_multi_image_v5(
    images: List[bytes],
    caption: str = "",
    generate_fn=None,
    exif_sidecar: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    V5 multi-image analysis pipeline (MIMI path).
    """
    report_hash = generate_report_hash(images[0] if images else None)

    system_prompt = load_mimi_v5_prompt()

    user_parts = []
    if caption:
        user_parts.append(
            f'RESIDENT CAPTION (optional, may be missing or unreliable):\n"{caption}"'
        )
    if exif_sidecar:
        exif_text = "EXIF SIDECAR (may be missing or partial):\n"
        for k, v in exif_sidecar.items():
            exif_text += f"- {k}: {v}\n"
        user_parts.append(exif_text)

    if generate_fn:
        try:
            raw_text = await generate_fn(system_prompt, images, user_parts)
        except Exception as e:
            logger.error(f"❌ V5 MIMI AI call failed: {e}")
            return build_v5_report_envelope([], [], report_hash, f"Multi-image analysis error: {e}")
    else:
        return build_v5_report_envelope([], [], report_hash, "No AI engine available")

    parsed = parse_v5_json(raw_text)
    if not parsed:
        return build_v5_report_envelope([], [], report_hash, "Failed to parse MIMI output", image_count=len(images))

    normalized = normalize_v5_output(parsed)
    normalized["schema_version"] = "v5-multi"

    scene_text = normalized.get("report_summary", "") + " " + caption
    post_result = run_post_processor(
        issues=normalized.get("issues", []),
        unknown_issues=normalized.get("unknown_issues", []),
        scene_text=scene_text,
    )

    envelope = build_v5_report_envelope(
        issues=post_result["issues"],
        unknown_issues=post_result["unknown_issues"],
        report_hash=report_hash,
        scene_description=normalized.get("report_summary", ""),
        image_count=len(images),
    )

    envelope["_post_processing"] = post_result["post_processing"]
    envelope["_post_processing"]["escalation_log"] = post_result["escalation_log"]
    envelope["_post_processing"]["confidence_band"] = post_result["confidence_band"]
    envelope["frames_analyzed"] = len(images)

    envelope = filter_report(envelope)
    envelope["_legacy_v3"] = map_v5_to_legacy(envelope)

    logger.info(
        f"✅ V5 MIMI Analysis complete: {len(envelope['issues'])} issues from {len(images)} images, "
        f"priority={envelope['final_priority']}"
    )

    return envelope
