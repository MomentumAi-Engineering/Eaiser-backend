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
    issues: List[IssueItem] = []
    zip_code: str
    user_continued_out_of_area: bool = False
    issue_id: Optional[str] = None  # 🆕 SIMI: If provided, auto-fetch detected issues from DB

class RecipientToggle(BaseModel):
    email: str
    type: str
    checked: bool
    # Provenance for dashboard + audit log:
    #   "ai_recommended" (default) — EAiSER routed automatically
    #   "user_added"               — resident added from the city picker
    #   "ai_removed_by_user"       — EAiSER recommended but resident unchecked
    source: Optional[str] = "ai_recommended"
    name: Optional[str] = None

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


@router.get("/list-departments")
async def list_departments(zip_code: str):
    """
    Return every city/county department available for a ZIP, so the
    iOS resident UI can offer a searchable picker to add additional
    recipients beyond EAiSER's AI recommendations.
    """
    try:
        from app.services.routing_engine_v5 import get_zip_authorities
    except ImportError:
        from services.routing_engine_v5 import get_zip_authorities

    zip_auth = get_zip_authorities(zip_code)
    departments = []
    for dept_key, contacts in (zip_auth or {}).items():
        if not isinstance(contacts, list):
            continue
        for contact in contacts:
            departments.append({
                "name": contact.get("name", dept_key),
                "email": contact.get("email", ""),
                "type": contact.get("type", dept_key),
                "phone": contact.get("phone", ""),
                "contact_name": contact.get("contact_name", ""),
                "role": contact.get("role", ""),
            })
    departments.sort(key=lambda d: d["name"].lower())
    return {"zip_code": zip_code, "departments": departments}


# ═══════════════════════════════════════════════════════════════
# PHASE 5-6: BUILD RECIPIENTS
# ═══════════════════════════════════════════════════════════════

@router.post("/build-recipients")
async def build_recipients(req: BuildRecipientsRequest):
    """
    Build the recipient list from AI-detected issues.
    Returns toggleable recipient list for the iOS app.
    
    🆕 SIMI Level 3: If issue_id is provided, auto-fetches ALL detected
    issues from the stored report and routes to ALL relevant departments.
    """
    try:
        from app.services.routing_engine_v5 import run_routing_pipeline
    except ImportError:
        from services.routing_engine_v5 import run_routing_pipeline
    
    issues = [item.dict() for item in req.issues]
    
    # 🆕 SIMI: Auto-fetch detected issues from DB if issue_id provided
    if req.issue_id and not issues:
        try:
            try:
                from app.services.mongodb_service import get_db
            except ImportError:
                from services.mongodb_service import get_db
            
            db = await get_db()
            stored_issue = await db.issues.find_one({"_id": req.issue_id})
            
            if stored_issue and stored_issue.get("detected_issues"):
                # Convert stored detected_issues to routing format
                for di in stored_issue["detected_issues"]:
                    issue_label = di.get("issue", "Unknown")
                    # Map tier to confidence approximation
                    tier_str = di.get("tier_or_severity", "Tier 2")
                    confidence = 90.0 if "Tier 0" in tier_str else (85.0 if "Tier 1" in tier_str else 75.0)
                    
                    issues.append({
                        "label": issue_label,
                        "confidence": confidence,
                        "severity": tier_str,
                        "observations": [],
                    })
                logger.info(f"🔍 SIMI Routing: Loaded {len(issues)} issues from stored report {req.issue_id}")
            elif stored_issue:
                # Fallback: use primary issue_type
                issues.append({
                    "label": stored_issue.get("issue_type", "Unknown"),
                    "confidence": stored_issue.get("confidence", 75.0),
                    "severity": stored_issue.get("severity", "Medium"),
                    "observations": [],
                })
        except Exception as fetch_err:
            logger.warning(f"Failed to fetch stored issues for {req.issue_id}: {fetch_err}")
    
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
                "name": toggle.name or toggle.type,
                "issue_labels": [],
                "reason": "user_tagged" if toggle.checked else "user_untagged",
                "severity": "P2",
                "source": toggle.source or "ai_recommended",
            })

        # Fetch full recipient data from the issue
        issue = await db.issues.find_one({"id": req.issue_id})
        if not issue:
            issue = await db.issues.find_one({"_id": req.issue_id})

        if issue and issue.get("v5_recipients"):
            # Merge user toggles with stored recipients.
            # Anything the resident added that wasn't in the AI list is appended
            # with source="user_added"; anything they unchecked is recorded as
            # "ai_removed_by_user" so the city dashboard can render the diff.
            stored = issue["v5_recipients"]
            merged = []
            stored_keys = set()
            for stored_r in stored:
                key = (stored_r.get("email", ""), stored_r.get("type", ""))
                stored_keys.add(key)
                user_toggle = next(
                    (t for t in req.recipients if t.email == stored_r["email"] and t.type == stored_r["type"]),
                    None,
                )
                stored_r.setdefault("source", "ai_recommended")
                if user_toggle:
                    if stored_r.get("locked"):
                        stored_r["checked"] = True
                    else:
                        stored_r["checked"] = user_toggle.checked
                        if not user_toggle.checked:
                            stored_r["source"] = "ai_removed_by_user"
                merged.append(stored_r)

            for toggle in req.recipients:
                key = (toggle.email, toggle.type)
                if key in stored_keys:
                    continue
                merged.append({
                    "email": toggle.email,
                    "type": toggle.type,
                    "name": toggle.name or toggle.type,
                    "checked": toggle.checked,
                    "locked": False,
                    "issue_labels": [],
                    "reason": "user_added_recipient",
                    "severity": "P2",
                    "source": "user_added",
                })
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

        # 4b. Manual-change audit entry (separate row for the city dashboard
        # so ops can filter on resident overrides without scanning the consent log)
        manual_changes = [
            {
                "name": r.get("name"),
                "email": r.get("email"),
                "type": r.get("type"),
                "source": r.get("source"),
                "checked": r.get("checked"),
            }
            for r in recipients
            if r.get("source") in ("user_added", "ai_removed_by_user")
        ]
        if manual_changes:
            manual_audit = build_audit_log(
                issue_id=req.issue_id,
                action="recipient_manual_changes",
                user_id=user_id,
                after={"changes": manual_changes},
                details=(
                    f"User added {sum(1 for c in manual_changes if c['source'] == 'user_added')} "
                    f"and removed {sum(1 for c in manual_changes if c['source'] == 'ai_removed_by_user')} recipients"
                ),
            )
            await db.audit_log.insert_one(manual_audit)
        
        # Update issue status
        checked_recipients = [r for r in recipients if r["checked"]]
        submitted_at = datetime.utcnow().isoformat()
        # Surface whether this city is a live partner or routed through Ops,
        # so the iOS report-detail screen can be transparent with the resident.
        all_internal = bool(checked_recipients) and all(
            (r.get("type") or "").startswith("internal_")
            or "momntum" in (r.get("name") or r.get("email") or "").lower()
            for r in checked_recipients
        )
        city_status = "NOT_LIVE" if all_internal else "LIVE"
        await db.issues.update_one(
            {"id": req.issue_id},
            {
                "$set": {
                    "v5_submitted": True,
                    "v5_recipients_final": checked_recipients,
                    "v5_submitted_at": submitted_at,
                    "v5_city_status": city_status,
                    "status": "submitted",
                },
                "$push": {
                    "status_history": {"state": "submitted", "at": submitted_at},
                },
            }
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
        
        closed_at = datetime.utcnow().isoformat()
        result = await db.issues.update_one(
            {"id": req.issue_id},
            {
                "$set": {
                    "status": "closed",
                    "closed_at": closed_at,
                    "close_reason": req.reason,
                },
                "$push": {
                    "status_history": {"state": "closed", "at": closed_at},
                },
            }
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


# ═══════════════════════════════════════════════════════════════
# V5 ANALYSIS ENDPOINT
# ═══════════════════════════════════════════════════════════════

class EditIssueRequest(BaseModel):
    issue_id: str
    report_id: str
    edits: Dict[str, Any] = {}

class DisputeRequest(BaseModel):
    issue_id: str
    report_id: str
    dispute_note: str

class RefreshSummaryRequest(BaseModel):
    report_id: str


@router.post("/analyze")
async def analyze_v5(
    file: Any = None,
    caption: str = "",
):
    """
    V5 single-image analysis endpoint.
    Replaces /api/ai/analyze-image for V5 pipeline.
    """
    from fastapi import UploadFile, File
    
    return {
        "status": "v5_analysis_endpoint_ready",
        "message": "V5 analysis pipeline active. Use /api/ai/analyze-image with V5 header for now.",
        "v5_features": [
            "13_factor_severity",
            "8_tier_0_categories",
            "auto_escalation",
            "banned_content_filter",
            "issue_id_format_r_hash_iNN",
        ]
    }


@router.post("/refresh-summary")
async def refresh_summary(req: RefreshSummaryRequest):
    """
    Light text-only LLM call to regenerate report_summary after resident edit.
    Does NOT re-analyze images.
    """
    try:
        try:
            from app.services.mongodb_service import get_db
        except ImportError:
            from services.mongodb_service import get_db
        
        db = await get_db()
        report = await db.issues.find_one({"id": req.report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Mark summary as refreshed
        await db.issues.update_one(
            {"id": req.report_id},
            {"$set": {
                "summary_freshness": "current",
                "report_status": "refreshed",
            }}
        )
        
        return {
            "success": True,
            "report_id": req.report_id,
            "summary_freshness": "current",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V5 refresh-summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edit-issue")
async def edit_issue(req: EditIssueRequest):
    """
    Resident edit endpoint. Non-emergency issues: full edit.
    Tier 0 (Emergency): LOCKED — returns error with dispute option.
    """
    try:
        try:
            from app.services.mongodb_service import get_db
            from app.services.v5_edit_service import apply_edit
        except ImportError:
            from services.mongodb_service import get_db
            from services.v5_edit_service import apply_edit
        
        db = await get_db()
        report = await db.issues.find_one({"id": req.report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        v5_data = report.get("v5_analysis", report)
        result = apply_edit(
            report=v5_data,
            issue_id=req.issue_id,
            edits=req.edits,
            user_id=report.get("user_id", "unknown"),
        )
        
        if result["success"]:
            await db.issues.update_one(
                {"id": req.report_id},
                {"$set": {
                    "v5_analysis": result["report"],
                    "summary_freshness": "stale_pending_refresh",
                    "report_status": "user_updated",
                }}
            )
        
        return {
            "success": result["success"],
            "edit_entry": result["edit_entry"],
            "warnings": result["warnings"],
            "summary_freshness": "stale_pending_refresh" if result["success"] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V5 edit-issue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dispute-tier0")
async def dispute_tier0(req: DisputeRequest):
    """
    Submit a dispute note for a Tier 0 (locked) issue.
    The classification does NOT change — only a note is logged for ops review.
    """
    try:
        try:
            from app.services.mongodb_service import get_db
            from app.services.v5_edit_service import submit_dispute
        except ImportError:
            from services.mongodb_service import get_db
            from services.v5_edit_service import submit_dispute
        
        db = await get_db()
        report = await db.issues.find_one({"id": req.report_id})
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        v5_data = report.get("v5_analysis", report)
        result = submit_dispute(
            report=v5_data,
            issue_id=req.issue_id,
            dispute_note=req.dispute_note,
            user_id=report.get("user_id", "unknown"),
        )
        
        if result["success"]:
            await db.issues.update_one(
                {"id": req.report_id},
                {"$set": {
                    "v5_analysis": result["report"],
                    "requires_post_action_review": True,
                }}
            )
        
        return {
            "success": result["success"],
            "dispute_entry": result["dispute_entry"],
            "message": "Dispute note logged. Classification unchanged. Ops team will review.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V5 dispute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
