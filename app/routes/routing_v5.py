"""
EAiSER V5 Routing API Routes
==============================
New API endpoints for the V5 routing pipeline.

Endpoints:
  POST /api/v5/check-zip       — Phase 2: ZIP Gate
  POST /api/v5/build-recipients — Phase 5-6: Build recipient list  
  POST /api/v5/submit           — Phase 8-11: Submit with consent
  POST /api/v5/close            — Phase 16: Close report
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v5", tags=["V5 Routing Engine"])

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class ZipCheckRequest(BaseModel):
    zip_code: str

class IssueItem(BaseModel):
    label: str
    confidence: float = 0
    severity: Optional[str] = None
    observations: Optional[List[str]] = []

class BuildRecipientsRequest(BaseModel):
    issues: List[IssueItem]
    zip_code: str
    user_continued_out_of_area: bool = False

class RecipientToggle(BaseModel):
    email: str
    type: str
    checked: bool

class SubmitWithConsentRequest(BaseModel):
    issue_id: str
    recipients: List[RecipientToggle]
    additional_notes: Optional[str] = ""

class CloseReportRequest(BaseModel):
    issue_id: str
    reason: Optional[str] = "resolved"


# ═══════════════════════════════════════════════════════════════
# PHASE 2: ZIP GATE
# ═══════════════════════════════════════════════════════════════

@router.post("/check-zip")
async def check_zip(req: ZipCheckRequest):
    """Check if zip code is in supported service area."""
    try:
        from app.services.routing_engine_v5 import check_zip_gate
    except ImportError:
        from services.routing_engine_v5 import check_zip_gate
    
    result = check_zip_gate(req.zip_code)
    return result


# ═══════════════════════════════════════════════════════════════
# PHASE 5-6: BUILD RECIPIENTS
# ═══════════════════════════════════════════════════════════════

@router.post("/build-recipients")
async def build_recipients(req: BuildRecipientsRequest):
    """
    Build the recipient list from AI-detected issues.
    Returns toggleable recipient list for the iOS app.
    """
    try:
        from app.services.routing_engine_v5 import run_routing_pipeline
    except ImportError:
        from services.routing_engine_v5 import run_routing_pipeline
    
    issues = [item.dict() for item in req.issues]
    result = run_routing_pipeline(
        issues=issues,
        zip_code=req.zip_code,
        user_continued_out_of_area=req.user_continued_out_of_area,
    )
    return result


# ═══════════════════════════════════════════════════════════════
# PHASE 8-11: SUBMIT WITH CONSENT
# ═══════════════════════════════════════════════════════════════

@router.post("/submit")
async def submit_with_consent(req: SubmitWithConsentRequest):
    """
    Submit report with user's consent (checked/unchecked states).
    Handles:
    - Phase 9: Hard guardrail (zero recipients check)
    - Phase 10: Persistence (consent_log, report_routings, report_ownership, audit_log)
    - Phase 11: Dispatch (immediate for P0, normal for others)
    """
    try:
        from app.services.routing_engine_v5 import (
            apply_hard_guardrail, build_consent_log,
            build_report_routings, build_report_ownership, build_audit_log
        )
        from app.services.mongodb_service import get_db
    except ImportError:
        from services.routing_engine_v5 import (
            apply_hard_guardrail, build_consent_log,
            build_report_routings, build_report_ownership, build_audit_log
        )
        from services.mongodb_service import get_db
    
    try:
        db = await get_db()
        
        # Build recipients with user toggles applied
        recipients = []
        for toggle in req.recipients:
            recipients.append({
                "email": toggle.email,
                "type": toggle.type,
                "checked": toggle.checked,
                "locked": False,  # Will be overridden below
                "name": toggle.type,
                "issue_labels": [],
                "reason": "user_tagged" if toggle.checked else "user_untagged",
                "severity": "P2",
            })
        
        # Fetch full recipient data from the issue
        issue = await db.issues.find_one({"id": req.issue_id})
        if not issue:
            issue = await db.issues.find_one({"_id": req.issue_id})
        
        if issue and issue.get("v5_recipients"):
            # Merge user toggles with stored recipients
            stored = issue["v5_recipients"]
            merged = []
            for stored_r in stored:
                user_toggle = next(
                    (t for t in req.recipients if t.email == stored_r["email"] and t.type == stored_r["type"]),
                    None
                )
                if user_toggle:
                    stored_r["checked"] = user_toggle.checked if not stored_r.get("locked") else True
                merged.append(stored_r)
            recipients = merged
        
        # Phase 9: Hard guardrail
        recipients = apply_hard_guardrail(recipients)
        
        # Phase 10: Persistence
        user_id = issue.get("user_id", "unknown") if issue else "unknown"
        
        # 1. Consent log
        consent = build_consent_log(req.issue_id, recipients, user_id)
        await db.consent_log.insert_one(consent)
        
        # 2. Report routings (only checked recipients)
        routings = build_report_routings(req.issue_id, recipients)
        if routings:
            await db.report_routings.insert_many(routings)
        
        # 3. Report ownership
        ownership = build_report_ownership(req.issue_id, recipients)
        if ownership:
            await db.report_ownership.insert_many(ownership)
        
        # 4. Audit log
        audit = build_audit_log(
            issue_id=req.issue_id,
            action="v5_submit_with_consent",
            user_id=user_id,
            after={"recipients": [r for r in recipients if r["checked"]]},
            details=f"Submitted to {sum(1 for r in recipients if r['checked'])} recipients"
        )
        await db.audit_log.insert_one(audit)
        
        # Update issue status
        checked_recipients = [r for r in recipients if r["checked"]]
        await db.issues.update_one(
            {"id": req.issue_id},
            {"$set": {
                "v5_submitted": True,
                "v5_recipients_final": checked_recipients,
                "v5_submitted_at": datetime.utcnow().isoformat(),
                "status": "submitted",
            }}
        )
        
        # Phase 11: Dispatch
        dispatched_count = 0
        locked_count = 0
        for r in checked_recipients:
            dispatched_count += 1
            if r.get("locked"):
                locked_count += 1
                # P0: Immediate alert (would trigger email_service here)
                logger.info(f"🔴 V5 Dispatch: IMMEDIATE alert to {r['name']} ({r['email']})")
            else:
                # Normal notification
                logger.info(f"📧 V5 Dispatch: Normal notification to {r['name']} ({r['email']})")
        
        return {
            "success": True,
            "issue_id": req.issue_id,
            "dispatched_to": dispatched_count,
            "locked_dispatched": locked_count,
            "recipients": [
                {"name": r["name"], "type": r["type"], "locked": r.get("locked", False)}
                for r in checked_recipients
            ],
            "message": f"Report sent to {dispatched_count} recipients",
        }
        
    except Exception as e:
        logger.error(f"V5 Submit error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# PHASE 16: CLOSE REPORT
# ═══════════════════════════════════════════════════════════════

@router.post("/close")
async def close_report(req: CloseReportRequest):
    """Close a report and set closed_at timestamp."""
    try:
        from app.services.mongodb_service import get_db
        from app.services.routing_engine_v5 import build_audit_log
    except ImportError:
        from services.mongodb_service import get_db
        from services.routing_engine_v5 import build_audit_log
    
    try:
        db = await get_db()
        
        result = await db.issues.update_one(
            {"id": req.issue_id},
            {"$set": {
                "status": "closed",
                "closed_at": datetime.utcnow().isoformat(),
                "close_reason": req.reason,
            }}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Audit log
        audit = build_audit_log(
            issue_id=req.issue_id,
            action="report_closed",
            user_id="system",
            details=f"Closed with reason: {req.reason}"
        )
        await db.audit_log.insert_one(audit)
        
        return {"success": True, "issue_id": req.issue_id, "closed_at": datetime.utcnow().isoformat()}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V5 Close error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
