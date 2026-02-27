

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

    def __init__(self, auto_dispatch_threshold: float = 75.0, fraud_reject_threshold: float = 85.0):
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

        # Decision logic based on User Requirements:
        # 1. Reject Case: Very low confidence or extreme fraud (spam/fakes)
        if fraud >= self.fraud_reject_threshold and confidence < 30:
             action = "reject"
             reasons.append("Critical fraud risk detected with extremely low confidence.")
             steps.append("Auto-screen out; do not notify.")
        
        # 2. Review Team Case: 
        # - Confidence <= 75%
        # - AI is confused or category is "other", "none", "unknown"
        # - Potential policy conflict or duplicates
        elif (
            confidence < self.auto_dispatch_threshold 
            or policy_conflict 
            or duplicates
            or issue_type in ["unknown", "other", "none", "general", "bonfire", "controlled_fire", "festival", "ceremony"]
        ):
            action = "route_to_review_team"
            if confidence < self.auto_dispatch_threshold:
                reasons.append(f"Confidence ({confidence}%) is below the direct authority notification threshold of {self.auto_dispatch_threshold}%.")
            if issue_type in ["unknown", "other", "none", "general"]:
                reasons.append(f"Issue category '{issue_type}' is ambiguous or unrecognized by our AI.")
            if policy_conflict:
                reasons.append("Environmental context suggests a potential false positive or policy-compliant activity.")
            if duplicates:
                reasons.append("Potential duplicate report detected in this area.")
            
            reasons.append("Report routed to EAiSER Admin Team for human verification before authority dispatch.")
            steps.append("EAiSER team must verify context and severity manually.")
            
        # 3. Auto Dispatch Case: Confidence > 75% + High Severity + Valid Metadata
        elif (
            confidence >= self.auto_dispatch_threshold
            and metadata_ok
            and not duplicates
        ):
            action = "auto_dispatch"
            reasons.append(f"High confidence ({confidence}%) report verified for direct authority notification.")
            steps.append("Directly dispatch alert to relevant municipal authorities via EAiSER Secure Portal.")
            
        # 4. Default Fallback
        else:
            action = "route_to_review_team"
            reasons.append("Defaulting to review team for additional safety verification.")
            steps.append("Manual review required.")

        return DispatchDecision(
            action=action,
            risk_score=risk,
            fraud_score=fraud,
            reasons=reasons,
            suggested_next_steps=steps,
        )