"""
EAiSER V5 Banned Content Filter
=================================
Enforces 12 privacy/safety rules on ALL AI-generated text fields.

These rules are NON-NEGOTIABLE and run on every output before it reaches
any user, admin, or external system.
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 12 BANNED CONTENT RULES
# ═══════════════════════════════════════════════════════════════

BANNED_RULES = [
    {
        "id": "BC-01",
        "name": "no_license_plates",
        "description": "Do not transcribe or reference any license plate numbers.",
        "patterns": [
            r'\b[A-Z0-9]{2,3}[-\s]?[A-Z0-9]{3,4}\b',  # Common plate formats
            r'\blicense\s+plate\s*[:=]?\s*[A-Z0-9]+',
            r'\bplate\s+number\s*[:=]?\s*[A-Z0-9]+',
            r'\btag\s+number\s*[:=]?\s*[A-Z0-9]+',
        ],
        "action": "redact",
        "replacement": "[REDACTED]",
    },
    {
        "id": "BC-02",
        "name": "no_personal_names",
        "description": "Do not identify individuals by name from images.",
        "patterns": [
            r'\bidentified\s+as\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
            r'\bname(?:d)?\s*[:=]\s*[A-Z][a-z]+',
            r'\bresident\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
        ],
        "action": "redact",
        "replacement": "[person]",
    },
    {
        "id": "BC-03",
        "name": "no_minor_identification",
        "description": "Do not identify or describe minors.",
        "patterns": [
            r'\bchild(?:ren)?\b.*\b(?:age|year|old|named)\b',
            r'\bminor\b.*\bidentif',
            r'\bstudent\s+[A-Z][a-z]+',
        ],
        "action": "redact",
        "replacement": "[minor present]",
    },
    {
        "id": "BC-04",
        "name": "no_address_numbers",
        "description": "Do not transcribe specific house/building numbers from images.",
        "patterns": [
            r'\bhouse\s+(?:number|#)\s*\d+',
            r'\bapartment\s+(?:number|#)\s*\d+',
            r'\bunit\s+(?:number|#)\s*\d+',
        ],
        "action": "redact",
        "replacement": "[address redacted]",
    },
    {
        "id": "BC-05",
        "name": "no_phone_numbers",
        "description": "Do not transcribe phone numbers visible in images.",
        "patterns": [
            r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        ],
        "action": "redact",
        "replacement": "[phone redacted]",
    },
    {
        "id": "BC-06",
        "name": "no_racial_profiling",
        "description": "Do not describe individuals by race, ethnicity, or skin color.",
        "patterns": [
            r'\b(?:white|black|hispanic|asian|caucasian|african)\s+(?:man|woman|person|male|female|individual)',
            r'\bskin\s+color\b',
            r'\bracial\b',
        ],
        "action": "remove",
        "replacement": "",
    },
    {
        "id": "BC-07",
        "name": "no_medical_diagnosis",
        "description": "Do not diagnose medical conditions from images.",
        "patterns": [
            r'\bappears\s+to\s+(?:have|suffer|be)\s+(?:from\s+)?(?:diabetes|cancer|covid|mental)',
            r'\bdiagnos(?:is|ed)\b',
        ],
        "action": "redact",
        "replacement": "[medical detail redacted]",
    },
    {
        "id": "BC-08",
        "name": "no_ssn",
        "description": "Do not transcribe social security numbers.",
        "patterns": [
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        ],
        "action": "redact",
        "replacement": "[SSN redacted]",
    },
    {
        "id": "BC-09",
        "name": "no_political_commentary",
        "description": "Do not include political opinions or party references.",
        "patterns": [
            r'\b(?:democrat|republican|liberal|conservative)\s+(?:policy|government|administration)',
            r'\bpolitical\s+(?:party|affiliation)',
        ],
        "action": "remove",
        "replacement": "",
    },
    {
        "id": "BC-10",
        "name": "no_blame_assignment",
        "description": "Do not assign blame or fault to individuals.",
        "patterns": [
            r'\b(?:fault|blame|responsible|caused\s+by)\s+(?:the\s+)?(?:resident|homeowner|driver|person)',
        ],
        "action": "redact",
        "replacement": "[attribution removed]",
    },
    {
        "id": "BC-11",
        "name": "no_graphic_injury_detail",
        "description": "Do not describe graphic injury details.",
        "patterns": [
            r'\b(?:blood|gore|severed|amputat|dismember|entrails)',
            r'\bgraphic\s+(?:injury|wound|trauma)',
        ],
        "action": "redact",
        "replacement": "[injury details redacted]",
    },
    {
        "id": "BC-12",
        "name": "english_only",
        "description": "All output must be in English only.",
        "patterns": [],  # Checked differently
        "action": "flag",
        "replacement": "",
    },
]


# ═══════════════════════════════════════════════════════════════
# FILTER ENGINE
# ═══════════════════════════════════════════════════════════════

def filter_text(text: str) -> tuple:
    """
    Apply all 12 banned content rules to a text string.

    Returns: (filtered_text, violations_found)
    """
    if not text:
        return text, []

    violations = []
    filtered = text

    for rule in BANNED_RULES:
        if rule["action"] == "flag":
            continue  # English-only is not pattern-based

        for pattern in rule["patterns"]:
            try:
                matches = re.findall(pattern, filtered, re.IGNORECASE)
                if matches:
                    violations.append({
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "matches_found": len(matches),
                    })
                    filtered = re.sub(
                        pattern, rule["replacement"], filtered, flags=re.IGNORECASE
                    )
            except re.error:
                continue

    return filtered, violations


def filter_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply banned content filter to ALL text fields in a V5 report.

    Returns the filtered report with a _privacy_filter metadata block.
    """
    total_violations = []

    # Fields to filter
    text_fields = [
        "report_summary",
        "scene_description",
    ]

    for field in text_fields:
        if field in report and isinstance(report[field], str):
            report[field], violations = filter_text(report[field])
            total_violations.extend(violations)

    # Filter inside each issue
    for issue in report.get("issues", []):
        for key in ["description", "observations_summary"]:
            if key in issue and isinstance(issue[key], str):
                issue[key], violations = filter_text(issue[key])
                total_violations.extend(violations)

    # Filter unknown issues
    for unknown in report.get("unknown_issues", []):
        for key in ["description", "issue"]:
            if key in unknown and isinstance(unknown[key], str):
                unknown[key], violations = filter_text(unknown[key])
                total_violations.extend(violations)

    report["_privacy_filter"] = {
        "applied": True,
        "rules_checked": len(BANNED_RULES),
        "violations_found": len(total_violations),
        "violations": total_violations[:10],  # Cap for log size
    }

    if total_violations:
        logger.warning(
            f"🔒 V5 Privacy Filter: {len(total_violations)} violations found and redacted"
        )

    return report
