"""
Modular summary builder for AI reports.
- Produces concise, human-readable summary using a strict template.
- OOP design for reusability and clean architecture.
"""

from typing import Any, Dict, List, Optional


class ReportSummaryBuilder:
    """Build a concise summary string from report payload fields.

    The template enforced:
    Our AI detected a {Issue_Type} in {City}, {State} (ZIP {Zip_Code}).
    The image shows {Short_Visual_Description}.
    Based on the location and context, this incident has been classified as {Priority_Label} due to {Risk_Tags}.

    Notes:
    - Missing City/State/ZIP will fall back to sensible defaults.
    - Risk tags are normalized and capped for readability.
    - Designed to be fast (synchronous) and safe.
    """

    DEFAULT_TEMPLATE = (
        "Our AI analyzed the image and identified a potential {Issue_Type} showing {Short_Visual_Description}. "
        "This issue is located at {City}, {State} (ZIP {Zip_Code})."
    )

    def __init__(self, template: Optional[str] = None, max_risk_tags: int = 5):
        # Allow custom template; keep output deterministic and short
        self.template = template or self.DEFAULT_TEMPLATE
        self.max_risk_tags = max_risk_tags

    def build(self, payload: Dict[str, Any]) -> str:
        """Create the summary text from provided payload.

        Expected keys:
        - Issue_Type, City, State, Zip_Code, Short_Visual_Description, Priority_Label, Risk_Tags
        Accepts alternate lowercase keys for resilience.
        """
        issue_type = self._val(payload, ["Issue_Type", "issue_type"], default="Issue")
        city = self._val(payload, ["City", "city"], default="Unknown City")
        state = self._format_state(self._val(payload, ["State", "state"], default="Unknown State"))
        zip_code = self._val(payload, ["Zip_Code", "zip_code"], default="N/A")
        svd = self._val(payload, ["Short_Visual_Description", "short_visual_description"], default="no clear visual description")
        priority = self._val(payload, ["Priority_Label", "priority_label", "priority"], default="Medium")
        risk_tags_str = self._risk_str(payload.get("Risk_Tags") or payload.get("risk_tags"))

        # Normalize formats (title case where applicable)
        summary = self.template.format(
            Issue_Type=self._title(issue_type, "Issue"),
            City=self._title(city, "Unknown City"),
            State=state,
            Zip_Code=zip_code or "N/A",
            Short_Visual_Description=svd,
            Priority_Label=self._title(priority, "Medium"),
            Risk_Tags=risk_tags_str or "general risk considerations",
        )
        return summary

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _val(self, data: Dict[str, Any], keys: List[str], default: str = "") -> str:
        """Safely extract a value from multiple possible keys."""
        for k in keys:
            v = data.get(k)
            if v is not None:
                return str(v).strip()
        return default

    def _title(self, s: Optional[str], default: str = "") -> str:
        """Title-case strings safely, with default fallback."""
        if not s:
            return default
        return str(s).strip().title()

    def _format_state(self, s: Optional[str]) -> str:
        """Format US state codes to uppercase if 2-letter; otherwise title-case."""
        if not s:
            return "Unknown State"
        s_clean = str(s).strip()
        return s_clean.upper() if len(s_clean) == 2 else s_clean.title()

    def _risk_str(self, tags: Any) -> str:
        """Normalize risk tags to a comma-separated string with a maximum length."""
        if not tags:
            return ""
        if isinstance(tags, str):
            return tags.strip()
        if isinstance(tags, (list, tuple)):
            cleaned = [str(t).strip() for t in tags if str(t).strip()]
            return ", ".join(cleaned[: self.max_risk_tags])
        return ""