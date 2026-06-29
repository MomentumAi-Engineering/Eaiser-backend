"""
Lightweight, defensive audit logging for security-sensitive admin actions.

Every mutating admin action (account provisioning, role/status changes,
password resets, contractor approvals, department + routing changes, city
tenant creation) records ONE immutable entry in the `audit_log` collection so
an IT Super Admin can review *who did what, when*.

Recording is best-effort: a logging failure must NEVER break the underlying
action — the write is wrapped in a broad try/except and only warns on failure.
"""
import logging
from datetime import datetime

from services.mongodb_service import get_db

logger = logging.getLogger(__name__)


async def record_audit(
    actor: dict,
    action: str,
    target: str = None,
    detail: str = None,
    city: str = None,
    meta: dict = None,
):
    """Write a single audit entry. Best-effort — swallows all errors.

    actor   — the authenticated user dict (email/role/name/org).
    action  — machine slug, e.g. "account_created", "contractor_approved".
    target  — what was acted on (email, contractor name, dept name, city…).
    detail  — short human-readable summary shown in the audit table.
    city    — tenant scope (defaults to the actor's org so the log is scopable).
    """
    try:
        db = await get_db()
        actor = actor or {}
        entry = {
            "action": action,
            "actor_email": actor.get("email"),
            "actor_role": actor.get("role"),
            "actor_name": actor.get("name"),
            "actor_type": actor.get("type"),
            "target": target,
            "detail": detail,
            "city": city or actor.get("org"),
            "meta": meta or {},
            "at": datetime.utcnow(),
        }
        await db["audit_log"].insert_one(entry)
    except Exception as e:  # never let auditing break the real action
        logger.warning(f"audit log write failed for action={action}: {e}")
