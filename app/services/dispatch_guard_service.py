

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DispatchDecision:
    """Final decision object jo UI/ops ke liye clear guidance deta hai."""
    action: str  # 'auto_dispatch' | 'hold_for_review' | 'reject'
    risk_score: float
    fraud_score: float
    reasons: List[str]
    suggested_next_steps: List[str]


class RiskScorer:
    """Risk score compute karta hai severity, priority, safety signals se."""

    def score(self, payload: Dict[str, Any]) -> float:
        # Base score from severity
        severity = (payload.get("severity") or "medium").lower()
        base = {"low": 35, "medium": 55, "high": 75, "urgent": 85}.get(severity, 55)

        # Public safety risk boost
        psr = (payload.get("public_safety_risk") or "").lower()
        if psr in ("medium", "high"):
            base += 10

        # Priority boost
        priority = (payload.get("priority") or "medium").lower()
        if priority in ("high", "urgent"):
            base += 8

        # Evidence signals
        evidence = payload.get("image_analysis") or {}
        if evidence.get("fire_ring_detected"):
            base += 3
        if evidence.get("sparks_visible"):
            base += 5
        if evidence.get("unattended"):
            base += 8

        return max(0, min(100, base))


class FraudScorer:
    """Fraud likelihood estimate karta hai confidence, duplicates, trust aur rate-burst se."""

    def score(self, payload: Dict[str, Any]) -> float:
        confidence = float(payload.get("ai_confidence_percent") or 0)
        reporter_trust = float(payload.get("reporter_trust_score") or 50)
        duplicates = bool(payload.get("is_duplicate"))
        rate_burst = bool(payload.get("rate_limit_breached"))
        metadata_ok = bool(payload.get("metadata_complete"))

        fraud = 0.0
        if confidence < 60:
            fraud += 30
        if not metadata_ok:
            fraud += 20
        if reporter_trust < 40:
            fraud += 15
        if duplicates:
            fraud += 25
        if rate_burst:
            fraud += 20

        return max(0, min(100, fraud))


class AuthorityDispatchGuard:
    """Guard orchestrates Risk & Fraud scorers to decide final action."""

    def __init__(self, auto_dispatch_threshold: float = 85, fraud_reject_threshold: float = 65):
        self.risk_scorer = RiskScorer()
        self.fraud_scorer = FraudScorer()
        self.auto_dispatch_threshold = auto_dispatch_threshold
        self.fraud_reject_threshold = fraud_reject_threshold

    def evaluate(self, payload: Dict[str, Any]) -> DispatchDecision:
        # Compute scores
        risk = self.risk_scorer.score(payload)
        fraud = self.fraud_scorer.score(payload)

        reasons: List[str] = []
        steps: List[str] = []

        # Simple signals
        confidence = float(payload.get("ai_confidence_percent") or 0)
        severity = (payload.get("severity") or "medium").lower()
        issue_type = str(payload.get("issue_type") or "").lower()
        metadata_ok = bool(payload.get("metadata_complete"))
        duplicates = bool(payload.get("is_duplicate"))
        policy_conflict = bool(payload.get("policy_conflict"))
        reporter_trust = float(payload.get("reporter_trust_score") or 50)

        # Decision logic
        # 1. Reject Case: EXTREMELY high fraud signals only
        # Raised threshold to 85 to prevent screening out legitimate but low-quality/duplicate tests
        if fraud >= 85 and confidence < 30:
             action = "reject"
             reasons.append("Critical fraud risk detected with low confidence.")
             if duplicates:
                 reasons.append("Duplicate report detected.")
             steps.append("Auto-screen out; do not notify.")
        
        # 2. Benign/Controlled Fire/Normal Image Case -> Route to Review Team
        # Now captures duplicates and low confidence reports instead of rejecting them
        # Changed confidence threshold from 60 to 70 as per user request
        # Also catches Unknown/Other/None issue types for manual review
        elif policy_conflict or confidence < 70 or severity == "low" or duplicates or issue_type in ["unknown", "other", "none", "general", "bonfire", "controlled_fire", "festival", "ceremony"]:
            action = "route_to_review_team"
            if policy_conflict:
                reasons.append("Policy context suggests controlled/allowed activity.")
            if confidence < 70:
                reasons.append(f"Confidence below threshold ({confidence}%).")
            if severity == "low":
                reasons.append("Low severity issue.")
            if duplicates:
                reasons.append("Potential duplicate report.")
            if issue_type in ["unknown", "other", "none", "general", "bonfire", "controlled_fire", "festival", "ceremony"]:
                reasons.append(f"Issue type '{issue_type}' routed for manual review.")
            steps.append("Route to internal review team for manual verification.")
            
        # 3. Auto Dispatch Case
        elif (
            confidence >= self.auto_dispatch_threshold
            and severity in ("high", "urgent")
            and metadata_ok
            and not duplicates
        ):
            action = "auto_dispatch"
            reasons.append("High confidence, severe issue, complete metadata, and no duplicates.")
            steps.append("Dispatch to authority; include concise summary and map link.")
            
        # 4. Default Fallback -> Hold for Review (General)
        else:
            action = "hold_for_review"
            if duplicates:
                reasons.append("Potential duplicate in same area/time window.")
            if reporter_trust < 40:
                reasons.append("Low reporter trust score.")
            if not metadata_ok:
                reasons.append("Incomplete metadata (address/coordinates/evidence).")
            steps.extend(
                [
                    "Request second photo/video or short confirmation.",
                    "Run duplicate check across recent nearby reports.",
                    "Verify geocoding; attach authoritative policy/burn-ban info.",
                    "Escalate to human reviewer if risk high (>70).",
                ]
            )

        return DispatchDecision(
            action=action,
            risk_score=risk,
            fraud_score=fraud,
            reasons=reasons,
            suggested_next_steps=steps,
        )