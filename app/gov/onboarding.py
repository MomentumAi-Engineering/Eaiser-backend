"""
EAiSER Civic — Post-Activation Onboarding

5-step wizard handlers. Persists progressive state so users can resume,
and finalizes the org configuration on completion.

Mounted at /api/gov/onboarding
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

try:
    from app.services.mongodb_service import get_db
except ImportError:
    from services.mongodb_service import get_db

try:
    from app.core.auth import get_current_user
except ImportError:
    try:
        from core.auth import get_current_user
    except ImportError:
        # Fallback shim — extract user from Bearer token manually if needed
        async def get_current_user(authorization: Optional[str] = Header(None)):
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Auth required")
            return {"email": "unknown"}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gov/onboarding", tags=["Onboarding"])

# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

class SaveStep(BaseModel):
    step: int
    data: Dict[str, Any]

class TeamInvite(BaseModel):
    email: str
    role: str = "ops"

class CompletePayload(BaseModel):
    publicName: Optional[str] = ""
    primaryColor: Optional[str] = "#D4A017"
    timezone: Optional[str] = "America/Chicago"
    departments: List[Dict[str, Any]] = []
    teamInvites: List[Dict[str, Any]] = []
    appName: Optional[str] = ""
    appShortDesc: Optional[str] = ""
    appPushColor: Optional[str] = "#D4A017"
    hasLogoUpload: bool = False
    logoDataUrl: Optional[str] = None

# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.post("/save")
async def save_step(payload: SaveStep, user=Depends(get_current_user)):
    """Persist intermediate onboarding state so the wizard can be resumed."""
    db = await get_db()
    email = (user.get("email") or user.get("sub", "")).lower()
    if not email:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    user_doc = await db["government_users"].find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    org_slug = user_doc.get("org_slug")
    if not org_slug:
        return {"success": False, "message": "No org assigned"}

    await db["organizations"].update_one(
        {"slug": org_slug},
        {"$set": {
            f"onboarding.step_{payload.step}": payload.data,
            "onboarding.current_step": payload.step,
            "updated_at": datetime.utcnow(),
        }},
        upsert=False,
    )
    return {"success": True}


@router.post("/complete")
async def complete_onboarding(
    payload: CompletePayload,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user),
):
    """
    Finalize onboarding: persist branding, departments, fire off team invites,
    mark org as fully onboarded and ready for daily use.
    """
    db = await get_db()
    email = (user.get("email") or user.get("sub", "")).lower()
    if not email:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    user_doc = await db["government_users"].find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    org_slug = user_doc.get("org_slug")
    if not org_slug:
        raise HTTPException(status_code=400, detail="No org assigned to this user")

    org_update = {
        "public_name": payload.publicName or "",
        "branding": {
            "primary_color": payload.primaryColor or "#D4A017",
            "logo_data_url": payload.logoDataUrl,
        },
        "timezone": payload.timezone or "America/Chicago",
        "citizen_app": {
            "name": payload.appName or "",
            "description": payload.appShortDesc or "",
            "accent_color": payload.appPushColor or "#D4A017",
        },
        "onboarding_complete": True,
        "onboarding_completed_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    await db["organizations"].update_one(
        {"slug": org_slug},
        {"$set": org_update},
    )

    # Replace departments wholesale
    if payload.departments:
        await db["org_departments"].delete_many({"org_slug": org_slug})
        dept_docs = []
        for d in payload.departments:
            if not d.get("name"):
                continue
            categories = d.get("categories", "")
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]
            dept_docs.append({
                "org_slug": org_slug,
                "name": d["name"],
                "color": d.get("color", "#D4A017"),
                "categories": categories,
                "created_at": datetime.utcnow(),
            })
        if dept_docs:
            await db["org_departments"].insert_many(dept_docs)

    # Fire off team invites (background)
    if payload.teamInvites:
        background_tasks.add_task(
            _process_team_invites,
            org_slug,
            payload.teamInvites,
            email,
        )

    return {
        "success": True,
        "org_slug": org_slug,
        "redirect_to": "/admin",
    }


@router.get("/state")
async def get_onboarding_state(user=Depends(get_current_user)):
    """Return current saved onboarding progress so the wizard can resume."""
    db = await get_db()
    email = (user.get("email") or user.get("sub", "")).lower()
    if not email:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    user_doc = await db["government_users"].find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    org_slug = user_doc.get("org_slug")
    if not org_slug:
        return {"step": 1, "data": {}}

    org = await db["organizations"].find_one({"slug": org_slug})
    if not org:
        return {"step": 1, "data": {}}

    onboarding = org.get("onboarding", {})
    current_step = onboarding.get("current_step", 1)
    if org.get("onboarding_complete"):
        return {"complete": True}

    saved = {k.replace("step_", ""): v for k, v in onboarding.items() if k.startswith("step_")}
    return {"step": current_step, "data": saved, "complete": False}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

async def _process_team_invites(org_slug: str, invites: List[Dict[str, Any]], invited_by: str):
    """Create pending user records + send activation emails for each invitee."""
    import secrets
    from datetime import timedelta

    try:
        try:
            from app.services.email_service import send_email
        except ImportError:
            from services.email_service import send_email

        db = await get_db()
        sent = 0

        for inv in invites:
            email = (inv.get("email") or "").lower().strip()
            role = inv.get("role", "ops")
            if not email:
                continue

            existing = await db["government_users"].find_one({"email": email})
            if existing:
                logger.info(f"Skipping invite for {email} — user already exists")
                continue

            token = secrets.token_urlsafe(32)
            now = datetime.utcnow()
            await db["government_users"].insert_one({
                "email": email,
                "first_name": email.split("@")[0].split(".")[0].title(),
                "last_name": "",
                "role": role,
                "status": "pending_activation",
                "activation_token": token,
                "activation_expires_at": now + timedelta(days=7),
                "org_slug": org_slug,
                "invited_by": invited_by,
                "invited_at": now,
                "created_at": now,
            })

            import os
            base = os.getenv("PORTAL_BASE_URL", "https://gov.eaiser.ai").rstrip("/")
            link = f"{base}/activate?token={token}"

            html = f"""
            <div style="font-family: 'Inter', sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 24px;">
                <h1 style="font-size: 22px; font-weight: 900;">EAiSER <span style="color: #D4A017;">CIVIC</span></h1>
                <p>You've been invited by {invited_by} to join EAiSER Civic as <strong>{role}</strong>.</p>
                <p style="text-align: center; margin: 32px 0;">
                    <a href="{link}" style="display: inline-block; background: #D4A017; color: #000; padding: 16px 32px; font-weight: 900; text-decoration: none; text-transform: uppercase; letter-spacing: 2px; font-size: 12px; border-radius: 4px;">Accept Invitation</a>
                </p>
                <p style="font-size: 11px; color: #888;">Link expires in 7 days.</p>
            </div>
            """
            result = send_email(email, "Invitation to join EAiSER Civic", html, f"Join EAiSER: {link}")
            import asyncio
            if asyncio.iscoroutine(result):
                await result
            sent += 1

        logger.info(f"Sent {sent} team invites for org {org_slug}")
    except Exception as e:
        logger.error(f"Failed processing team invites for {org_slug}: {e}")
