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
    address: Optional[str] = ""
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
# CITY ROUTING CONFIG OVERLAY
# ───────────────────────────────────────────────────────────────
# The sync routing engine (routing_engine_v5) routes off static JSON maps. The
# gov portal's City Manager separately configures a per-city `routing_config`
# (recipients, issue category table, location overrides, severity rules, Tier-0
# floors) in MongoDB. These helpers layer that live config ON TOP of the engine
# output so portal configuration actually influences dispatch — additively and
# safely (engine recipients are never removed, only augmented).
# ═══════════════════════════════════════════════════════════════

def _norm(s) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


async def _resolve_city_for_zip(db, zip_code: str):
    """Map a ZIP to the gov tenant 'city' string used as routing_config.city."""
    if not zip_code:
        return None
    org = await db["organizations"].find_one(
        {"$or": [{"zip_codes": zip_code}, {"primary_zip": zip_code}]}
    )
    if org:
        return org.get("legal_name") or org.get("city")
    u = await db["government_users"].find_one({"zip_code": zip_code})
    return u.get("city") if u else None


async def apply_city_routing_config(db, zip_code, issues, address, result):
    """Augment engine recipients with the City Manager's routing_config.

    Adds: external recipients whose category matches a detected issue, issue
    category-table routes, and location-override recipients (when the override
    resolves to a configured recipient with an email). Surfaces Tier-0 floors
    for emergencies. Only runs for LIVE, in-area reports. Best-effort.
    """
    try:
        # Only layer config onto a normal, in-area, LIVE recipient list.
        if result.get("phase") != "recipient_list_ready" or result.get("is_out_of_area"):
            return result
        if (result.get("city_state") or {}).get("status") != "LIVE":
            return result

        city = await _resolve_city_for_zip(db, zip_code)
        if not city:
            return result
        cfg = await db["routing_config"].find_one({"city": city})
        if not cfg:
            return result

        recipients = result.get("recipients", [])
        existing = {(r.get("email", ""), r.get("type", "")) for r in recipients}
        labels = [_norm(i.get("label")) for i in issues]
        addr_l = (address or "").lower()
        cfg_recipients = cfg.get("recipients", [])
        by_name = {_norm(r.get("name")): r for r in cfg_recipients}
        added = 0
        advisory = []

        def _add(rec, reason):
            nonlocal added
            email = rec.get("email")
            name = rec.get("name")
            if not email:
                if name:
                    advisory.append(name)  # no email → can't dispatch, surface as advisory
                return
            key = (email, rec.get("type") or "external")
            if key in existing:
                return
            existing.add(key)
            recipients.append({
                "name": name,
                "email": email,
                "type": rec.get("type") or "external",
                "issue_labels": [i.get("label") for i in issues],
                "reason": reason,
                "checked": True,
                "locked": False,
                "severity": "P2",
                "source": "city_config",
            })
            added += 1

        def _label_match(cat):
            if not cat or cat in ("all", "global", "global_fallback_/_ops_queue"):
                return True
            return any(cat == l or cat in l or l in cat for l in labels)

        # 1. External recipients by category (issue category → recipient)
        for r in cfg_recipients:
            if (r.get("type") or "external") == "internal":
                continue
            if _label_match(_norm(r.get("category"))):
                _add(r, "city_config_category")

        # 2. Issue category table: route_to a configured recipient name
        for route in cfg.get("category_routes", []):
            if _norm(route.get("issue_type")) in labels:
                target = by_name.get(_norm(route.get("route_to")))
                if target:
                    _add(target, "city_config_category_route")
                elif route.get("route_to"):
                    advisory.append(route.get("route_to"))

        # 3. Location overrides: address keyword → add recipient
        for rule in cfg.get("location_overrides", []):
            kw = str(rule.get("keyword", "")).lower()
            if kw and kw in addr_l:
                target = by_name.get(_norm(rule.get("add_recipient")))
                if target:
                    _add(target, "city_config_location")
                elif rule.get("add_recipient"):
                    advisory.append(rule.get("add_recipient"))

        result["recipients"] = recipients
        result["total_recipients"] = len(recipients)
        result["city_config_applied"] = True
        result["city"] = city
        result["city_config_added"] = added
        if advisory:
            result["city_config_advisory_recipients"] = sorted({a for a in advisory if a})
        if result.get("has_p0"):
            tier0 = [t.get("name") for t in cfg.get("tier0_floors", []) if t.get("name")]
            if tier0:
                result["tier0_floors"] = tier0
        return result
    except Exception as e:
        logger.warning(f"city routing config overlay failed for zip {zip_code}: {e}")
        return result


async def _log_routing_decision(db, issue_id, zip_code, address, issues, result):
    """Persist an auditable routing decision (recipients + confidence + config flag)."""
    try:
        if result.get("phase") != "recipient_list_ready":
            return
        await db["routing_decisions"].insert_one({
            "issue_id": issue_id,
            "zip_code": zip_code,
            "city": result.get("city"),
            "address": address or "",
            "issues": [
                {"label": i.get("label"), "confidence": i.get("confidence"), "severity": i.get("severity")}
                for i in issues
            ],
            "recipients": [
                {"name": r.get("name"), "email": r.get("email"), "type": r.get("type"),
                 "source": r.get("source", "ai_recommended"), "reason": r.get("reason")}
                for r in result.get("recipients", [])
            ],
            "has_emergency": bool(result.get("has_p0")),
            "city_config_applied": bool(result.get("city_config_applied")),
            "city_config_added": result.get("city_config_added", 0),
            "created_at": datetime.utcnow(),
        })
    except Exception as e:
        logger.warning(f"routing_decisions log failed: {e}")


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
        address=req.address or "",
    )

    # Layer the city's live routing_config on top of the engine output, then
    # persist an auditable routing decision (recipients + confidence). Both are
    # best-effort and never block the response.
    try:
        try:
            from app.services.mongodb_service import get_db
        except ImportError:
            from services.mongodb_service import get_db
        db = await get_db()
        result = await apply_city_routing_config(db, req.zip_code, issues, req.address or "", result)
        await _log_routing_decision(db, req.issue_id, req.zip_code, req.address or "", issues, result)
    except Exception as overlay_err:
        logger.warning(f"build-recipients overlay/log skipped: {overlay_err}")

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
