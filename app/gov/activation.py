"""
EAiSER Civic — Account Activation

Handles magic-link account activation for newly-provisioned admin users
and team invitees. Tokens are single-use, time-bound, cryptographically secure.

Mounted at /api/gov/activation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime, timedelta
import secrets
import logging

try:
    from app.services.mongodb_service import get_db
except ImportError:
    from services.mongodb_service import get_db

try:
    from app.utils.security import get_password_hash, create_access_token
except ImportError:
    from utils.security import get_password_hash, create_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gov/activation", tags=["Account Activation"])

# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

class CompleteActivation(BaseModel):
    token: str
    password: str = Field(..., min_length=12)
    enable_mfa: bool = True

class InviteRequest(BaseModel):
    email: EmailStr
    role: str = "ops"

class BulkInvite(BaseModel):
    invites: List[InviteRequest]

# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.get("/verify")
async def verify_activation_token(token: str):
    """
    Verify a magic-link activation token. Returns user/org context for the activation page.
    Does NOT consume the token — completion call does.
    """
    if not token:
        raise HTTPException(status_code=400, detail="Token required")

    db = await get_db()
    user = await db["government_users"].find_one({"activation_token": token})

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or already-used activation link")

    if user.get("status") == "active":
        raise HTTPException(status_code=400, detail="Account already activated. Please log in.")

    expires_at = user.get("activation_expires_at")
    if expires_at and expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="This activation link has expired. Please contact support.")

    # Look up org for context
    org = None
    org_id = user.get("org_id")
    if org_id:
        try:
            from bson import ObjectId
            org = await db["organizations"].find_one({"_id": ObjectId(org_id)})
        except Exception:
            org = None

    return {
        "email": user.get("email"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "role": user.get("role"),
        "org_name": org.get("legal_name") if org else user.get("city"),
        "org_slug": user.get("org_slug"),
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


@router.post("/complete")
async def complete_activation(payload: CompleteActivation):
    """
    Finalize account activation. Consumes token, sets password, issues access JWT.
    """
    if not _validate_password_strength(payload.password):
        raise HTTPException(status_code=400, detail="Password does not meet strength requirements")

    db = await get_db()
    user = await db["government_users"].find_one({"activation_token": payload.token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or already-used activation link")

    if user.get("status") == "active":
        raise HTTPException(status_code=400, detail="Account already activated. Please log in.")

    expires_at = user.get("activation_expires_at")
    if expires_at and expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Activation link has expired")

    now = datetime.utcnow()
    update_doc = {
        "$set": {
            "hashed_password": get_password_hash(payload.password),
            "status": "active",
            "is_active": True,
            "activated_at": now,
            "mfa_enabled": payload.enable_mfa,
            "require_mfa_setup": payload.enable_mfa,
            "updated_at": now,
        },
        "$unset": {
            "activation_token": "",
            "activation_expires_at": "",
        }
    }

    await db["government_users"].update_one({"_id": user["_id"]}, update_doc)

    # Issue access token so user goes straight into onboarding
    access_token = create_access_token(
        data={
            "sub": user["email"],
            "type": "gov_portal",
            "role": user.get("role", "super_admin"),
            "org": user.get("org_slug"),
            "dept": user.get("department", ""),
        }
    )

    user_payload = {
        "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
        "email": user.get("email"),
        "role": user.get("role"),
        "org": user.get("org_slug"),
        "city": user.get("city"),
        "initials": "".join([p[0].upper() for p in [user.get("first_name", "U"), user.get("last_name", "")] if p])[:2],
    }

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_payload,
    }


@router.post("/resend")
async def resend_activation(email: EmailStr, background_tasks: BackgroundTasks):
    """
    Regenerate and resend activation link for users who lost / expired their original.
    """
    db = await get_db()
    user = await db["government_users"].find_one({"email": email.lower(), "status": "pending_activation"})
    if not user:
        # Don't leak existence — return generic success
        return {"success": True, "message": "If an account exists, an activation email has been sent."}

    new_token = secrets.token_urlsafe(32)
    new_expiry = datetime.utcnow() + timedelta(days=7)

    await db["government_users"].update_one(
        {"_id": user["_id"]},
        {"$set": {"activation_token": new_token, "activation_expires_at": new_expiry}}
    )

    background_tasks.add_task(
        _send_activation_email,
        email.lower(),
        user.get("first_name", "there"),
        user.get("city", ""),
        new_token,
    )

    return {"success": True, "message": "Activation email sent."}


@router.post("/invite")
async def send_team_invites(payload: BulkInvite, background_tasks: BackgroundTasks):
    """
    Invite team members to an existing org. Generates activation tokens
    and sends magic-link emails. Authentication expected via Authorization header
    (validation done in main pipeline; for now we trust the requesting org context).
    """
    db = await get_db()
    sent = []
    skipped = []

    for inv in payload.invites:
        email = inv.email.lower()
        if not email:
            continue

        existing = await db["government_users"].find_one({"email": email})
        if existing:
            skipped.append({"email": email, "reason": "already exists"})
            continue

        token = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        user_doc = {
            "email": email,
            "first_name": email.split("@")[0].split(".")[0].title(),
            "last_name": "",
            "role": inv.role,
            "status": "pending_activation",
            "activation_token": token,
            "activation_expires_at": now + timedelta(days=7),
            "invited_at": now,
            "created_at": now,
        }
        await db["government_users"].insert_one(user_doc)

        background_tasks.add_task(
            _send_invite_email,
            email,
            inv.role,
            token,
        )
        sent.append(email)

    return {"success": True, "sent": sent, "skipped": skipped}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _validate_password_strength(password: str) -> bool:
    import re
    if len(password) < 12:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    if not re.search(r"[^A-Za-z0-9]", password):
        return False
    return True


def _build_activation_link(token: str) -> str:
    import os
    base = os.getenv("PORTAL_BASE_URL", "https://gov.eaiser.ai").rstrip("/")
    return f"{base}/activate?token={token}"


async def _send_activation_email(to: str, first_name: str, city: str, token: str):
    try:
        try:
            from app.services.email_service import send_email
        except ImportError:
            from services.email_service import send_email

        link = _build_activation_link(token)
        subject = "Your EAiSER activation link"
        html = f"""
        <div style="font-family: 'Inter', sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 24px;">
            <h1 style="font-size: 22px; font-weight: 900;">EAiSER <span style="color: #D4A017;">CIVIC</span></h1>
            <p>Hi {first_name},</p>
            <p>Here is your new activation link. Valid for 7 days:</p>
            <p style="text-align: center; margin: 32px 0;">
                <a href="{link}" style="display: inline-block; background: #D4A017; color: #000; padding: 16px 32px; font-weight: 900; text-decoration: none; text-transform: uppercase; letter-spacing: 2px; font-size: 12px; border-radius: 4px;">Activate Account</a>
            </p>
            <p style="font-size: 11px; color: #888;">If the button doesn't work: <a href="{link}">{link}</a></p>
        </div>
        """
        await _maybe_await(send_email, to, subject, html, f"Activate your EAiSER account: {link}")
    except Exception as e:
        logger.error(f"Failed to send activation email to {to}: {e}")


async def _send_invite_email(to: str, role: str, token: str):
    try:
        try:
            from app.services.email_service import send_email
        except ImportError:
            from services.email_service import send_email

        link = _build_activation_link(token)
        subject = f"You've been invited to EAiSER Civic ({role.title()})"
        html = f"""
        <div style="font-family: 'Inter', sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 24px;">
            <h1 style="font-size: 22px; font-weight: 900;">EAiSER <span style="color: #D4A017;">CIVIC</span></h1>
            <p>You've been invited to join EAiSER Civic as <strong>{role}</strong>.</p>
            <p>Click below to activate your account and set your password:</p>
            <p style="text-align: center; margin: 32px 0;">
                <a href="{link}" style="display: inline-block; background: #D4A017; color: #000; padding: 16px 32px; font-weight: 900; text-decoration: none; text-transform: uppercase; letter-spacing: 2px; font-size: 12px; border-radius: 4px;">Accept Invitation</a>
            </p>
            <p style="font-size: 11px; color: #888;">Link expires in 7 days. If it doesn't work: <a href="{link}">{link}</a></p>
        </div>
        """
        await _maybe_await(send_email, to, subject, html, f"Join EAiSER Civic: {link}")
    except Exception as e:
        logger.error(f"Failed to send invite email to {to}: {e}")


async def _maybe_await(fn, *args, **kwargs):
    import asyncio
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result
