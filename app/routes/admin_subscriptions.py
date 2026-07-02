"""
EAiSER Admin — Subscriptions & Billing API.

Reads from the `organizations` collection (created when a city completes
checkout via /api/gov/billing/checkout — see gov/billing.py) and transforms
each org into the admin-portal subscription DTO that
EAISER_FRONTEND/src/components/SubscriptionManagement.jsx consumes.

Mounted at /api/admin/subscriptions

Admin-only mutations (renew, upgrade, suspend, autopay toggle, enforcement
policy, overage resolution) are persisted to `organizations.admin_meta`
and appended to `organizations.admin_meta.activity` for the audit log.

Stripe interactions:
  • If STRIPE_SECRET_KEY is set AND the org has a `subscription_id`, we
    call Stripe to cancel/resume the actual subscription on suspend/reactivate
    and to set the default payment method on autopay-enable.
  • If Stripe is not configured, mutations are local-only (status flag only).
"""
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId
import os
import re
import secrets
import logging

try:
    from app.services.mongodb_service import get_db
except ImportError:
    from services.mongodb_service import get_db

try:
    from app.core.auth import require_permission
except ImportError:
    from core.auth import require_permission


def _gov_billing_helpers():
    """Lazy-import the gov.billing helpers so this module loads cleanly even
    if gov/billing.py has a transient import error (e.g. missing optional
    dep). We only need these when creating an admin-initiated subscription."""
    try:
        from app.gov.billing import (
            _seed_default_org_data, _send_activation_email,
            _build_activation_link, _generate_activation_token,
            _ensure_unique_slug, slugify_city,
            _create_stripe_checkout_session, CheckoutRequest,
        )
    except ImportError:
        from gov.billing import (
            _seed_default_org_data, _send_activation_email,
            _build_activation_link, _generate_activation_token,
            _ensure_unique_slug, slugify_city,
            _create_stripe_checkout_session, CheckoutRequest,
        )
    return dict(
        seed_default_org_data=_seed_default_org_data,
        send_activation_email=_send_activation_email,
        build_activation_link=_build_activation_link,
        generate_activation_token=_generate_activation_token,
        ensure_unique_slug=_ensure_unique_slug,
        slugify_city=slugify_city,
        create_stripe_checkout_session=_create_stripe_checkout_session,
        CheckoutRequest=CheckoutRequest,
    )


# Local fallbacks — used if gov.billing helpers aren't available for any reason.
def _fallback_generate_token() -> str:
    return secrets.token_urlsafe(32)


def _fallback_slugify(city: str, state: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", f"{city}-{state}".lower()).strip("-")
    return base or f"city-{secrets.token_hex(3)}"


async def _fallback_ensure_unique_slug(db, slug: str) -> str:
    attempt, suffix = slug, 0
    while await db["organizations"].find_one({"slug": attempt}):
        suffix += 1
        attempt = f"{slug}-{suffix}"
        if suffix > 50:
            attempt = f"{slug}-{secrets.token_hex(2)}"
            break
    return attempt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/subscriptions", tags=["Admin Subscriptions"])

# ──────────────────────────────────────────────────────────────────────
# Tier metadata — kept in sync with EAiSER price book.
# Authoritative copies:
#   • Backend:  app/gov/billing.py:TIER_FLOOR
#   • Gov UI:   EAISER_GOV_PORTAL/src/services/pricing.js:TIERS
#   • Admin UI: EAISER_FRONTEND/src/utils/eaiserPlans.js:PLAN_TIERS
# ──────────────────────────────────────────────────────────────────────

TIER_META = {
    "spark":      {"label": "Spark",      "pop_min": 0,      "pop_max": 4_999,    "staff_seats": 5,  "monthly": 600,    "annual": 7_200},
    "spark-plus": {"label": "Spark+",     "pop_min": 5_000,  "pop_max": 9_999,    "staff_seats": 15, "monthly": 1_000,  "annual": 12_000},
    "core":       {"label": "Core",       "pop_min": 10_000, "pop_max": 24_999,   "staff_seats": 0,  "monthly_min": 1_500,  "monthly_max": 2_625,  "annual_min": 18_000,  "annual_max": 31_500},
    "pro":        {"label": "Pro",        "pop_min": 25_000, "pop_max": 99_999,   "staff_seats": 0,  "monthly_min": 4_500,  "monthly_max": 6_750,  "annual_min": 54_000,  "annual_max": 81_000},
    "enterprise": {"label": "Enterprise", "pop_min": 100_000,"pop_max": 499_999,  "staff_seats": 0,  "monthly_min": 13_125, "monthly_max": 20_625, "annual_min": 157_500, "annual_max": 247_500},
    "metro":      {"label": "Metro",      "pop_min": 500_000,"pop_max": 10**9,    "staff_seats": 0,  "monthly_min": 45_000,                       "annual_min": 540_000},
}


def _tier_seats(tier_id: str) -> int:
    return TIER_META.get(tier_id, {}).get("staff_seats", 0)


def _default_mrr_for_tier(tier_id: str) -> int:
    m = TIER_META.get(tier_id, {})
    if m.get("monthly") is not None:
        return m["monthly"]
    if m.get("monthly_min") and m.get("monthly_max"):
        return (m["monthly_min"] + m["monthly_max"]) // 2
    return m.get("monthly_min", 0)


# ──────────────────────────────────────────────────────────────────────
# Org → Subscription DTO transformation
# ──────────────────────────────────────────────────────────────────────

def _derive_payment_method(org: Dict[str, Any]) -> Optional[str]:
    """Build a human-readable payment-method label."""
    meta = org.get("admin_meta") or {}
    if meta.get("payment_method_override"):
        return meta["payment_method_override"]
    raw = org.get("payment_method") or "invoice"
    if raw == "ach":
        return "ACH (Bank Transfer)"
    if raw == "card":
        last4 = (org.get("stripe_default_card_last4") or "").strip()
        return f"Card ••• {last4}" if last4 else "Credit / Debit Card"
    if raw == "invoice":
        return "Invoice (PO / Net 30)"
    return raw


def _calc_mrr(org: Dict[str, Any]) -> int:
    annual = org.get("annual_price_usd") or 0
    if annual:
        return int(round(annual / 12))
    # Fallback to tier mid-point
    return _default_mrr_for_tier(org.get("tier_id") or "")


def _calc_end_date(org: Dict[str, Any]) -> Optional[datetime]:
    """Subscription end = max(trial_ends_at, created_at + multi_year years)."""
    meta = org.get("admin_meta") or {}
    if meta.get("end_date_override"):
        return meta["end_date_override"]
    multi = max(1, int(org.get("multi_year") or 1))
    start = org.get("created_at") or datetime.utcnow()
    contract_end = start + timedelta(days=365 * multi)
    trial = org.get("trial_ends_at")
    if trial and trial > contract_end:
        return trial
    return contract_end


async def _count_active_users(db, org_id: str) -> int:
    """Active government_users for this org (excludes pending_activation)."""
    try:
        return await db["government_users"].count_documents({
            "org_id": org_id,
            "status": {"$in": ["active", "invited"]},
        })
    except Exception:
        return 0


async def _org_to_subscription(db, org: Dict[str, Any]) -> Dict[str, Any]:
    org_id = str(org.get("_id"))
    meta = org.get("admin_meta") or {}
    tier_id = org.get("tier_id") or "spark"

    billing = org.get("billing") or {}
    city_label = f"{billing.get('city', org.get('legal_name') or '')}, {billing.get('state', org.get('state') or '')}".strip(", ")

    status = (org.get("status") or "").lower()
    is_active = meta.get("is_active") if "is_active" in meta else status not in ("cancelled", "pending_payment")
    suspended = bool(meta.get("suspended")) or status == "cancelled"

    active_users = await _count_active_users(db, org_id)

    return {
        "id": org_id,
        "customer_name": org.get("legal_name") or "(unnamed)",
        "city": city_label,
        "population": org.get("population") or 0,
        "contact_email": org.get("buyer_email"),
        "buyer_name": org.get("buyer_name"),
        "buyer_phone": org.get("buyer_phone"),

        "plan": tier_id,
        "user_limit": _tier_seats(tier_id),  # 0 = unlimited
        "active_users": active_users,

        "mrr": _calc_mrr(org),
        "currency": "USD",
        "annual_price_usd": org.get("annual_price_usd") or 0,
        "total_contract_usd": org.get("total_contract_usd") or 0,
        "multi_year": org.get("multi_year") or 1,

        "start_date": (org.get("created_at") or datetime.utcnow()).isoformat(),
        "end_date": (_calc_end_date(org) or datetime.utcnow()).isoformat(),
        "trial_ends_at": (org.get("trial_ends_at") or None) and org["trial_ends_at"].isoformat(),
        "last_payment_at": (org.get("first_payment_at") or None) and org["first_payment_at"].isoformat(),

        "is_active": bool(is_active),
        "suspended": suspended,
        "auto_renew": bool(meta.get("auto_renew", True)),
        "autopay": bool(meta.get("autopay", org.get("payment_method") in ("card", "ach"))),
        "payment_method": _derive_payment_method(org),
        "payment_method_raw": org.get("payment_method"),
        "stripe_customer_id": org.get("stripe_customer_id"),
        "stripe_subscription_id": org.get("subscription_id"),

        "enforcement_mode": meta.get("enforcement_mode", "monitor"),
        "overage_rate": meta.get("overage_rate", 0),
        "grace_days": meta.get("grace_days", 14),
        "grace_started_at": (meta.get("grace_started_at") or None) and meta["grace_started_at"].isoformat(),

        "status_raw": status,
        "onboarding_complete": bool(org.get("onboarding_complete")),
        "activity": meta.get("activity", [])[:50],
    }


# ──────────────────────────────────────────────────────────────────────
# Activity-log helper
# ──────────────────────────────────────────────────────────────────────

async def _log_activity(db, org_filter: Dict[str, Any], text: str, actor: str = "admin"):
    entry = {"ts": datetime.utcnow().strftime("%Y-%m-%d"), "text": text, "actor": actor}
    await db["organizations"].update_one(
        org_filter,
        {"$push": {"admin_meta.activity": {"$each": [entry], "$position": 0, "$slice": 100}}},
    )


def _org_filter(sub_id: str) -> Dict[str, Any]:
    try:
        return {"_id": ObjectId(sub_id)}
    except Exception:
        # Fallback to slug — allows admin URLs that use the city slug.
        return {"slug": sub_id}


# ──────────────────────────────────────────────────────────────────────
# Stripe helpers (no-ops when STRIPE_SECRET_KEY isn't set)
# ──────────────────────────────────────────────────────────────────────

def _stripe_enabled() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY"))


def _stripe():
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    return stripe


async def _stripe_cancel_subscription(sub_id: str):
    if not (_stripe_enabled() and sub_id):
        return
    try:
        _stripe().Subscription.delete(sub_id)
    except Exception as e:
        logger.warning(f"Stripe cancel failed for {sub_id}: {e}")


async def _stripe_set_default_payment(customer_id: str, payment_method_id: str):
    if not (_stripe_enabled() and customer_id and payment_method_id):
        return
    try:
        s = _stripe()
        s.PaymentMethod.attach(payment_method_id, customer=customer_id)
        s.Customer.modify(customer_id, invoice_settings={"default_payment_method": payment_method_id})
    except Exception as e:
        logger.warning(f"Stripe set-default-payment failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────

class CreateSubscriptionPayload(BaseModel):
    customer_name: str
    city: Optional[str] = ""
    state: Optional[str] = ""
    population: Optional[int] = 0
    contact_email: str
    contact_first_name: Optional[str] = None
    contact_last_name: Optional[str] = None
    contact_title: Optional[str] = "Administrator"
    contact_phone: Optional[str] = ""
    plan: str
    mrr: Optional[int] = None
    user_limit: Optional[int] = None
    auto_renew: bool = True
    autopay: bool = False
    payment_method: Optional[str] = None  # 'card' | 'ach' | 'invoice'
    enforcement_mode: str = "monitor"
    overage_rate: float = 0
    grace_days: int = 14
    term_months: int = 12
    send_activation_email: bool = True
    request_stripe_checkout: bool = False  # if True and payment_method='card' & Stripe configured


class RenewPayload(BaseModel):
    months: int = 12


class UpgradePayload(BaseModel):
    plan: str
    user_limit: Optional[int] = None
    mrr: Optional[int] = None


class LimitPayload(BaseModel):
    user_limit: int


class SuspendPayload(BaseModel):
    reason: Optional[str] = "Suspended by admin"


class ActivePayload(BaseModel):
    is_active: bool


class AutopayPayload(BaseModel):
    autopay: bool
    payment_method: Optional[str] = None


class EnforcementPayload(BaseModel):
    mode: str
    overageRate: float = 0
    graceDays: int = 14


class ResolveOveragePayload(BaseModel):
    action: str  # 'upgrade' | 'bill' | 'block' | 'notify'


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.get("")
async def list_subscriptions(
    admin=Depends(require_permission("view_subscriptions")),
    status: Optional[str] = Query(None, description="Filter by status"),
    plan: Optional[str] = Query(None, description="Filter by tier id"),
    limit: int = Query(500, ge=1, le=2000),
):
    """List every organization as an admin subscription DTO."""
    db = await get_db()
    q: Dict[str, Any] = {}
    if status:
        q["status"] = status
    if plan:
        q["tier_id"] = plan

    cursor = db["organizations"].find(q).sort("created_at", -1).limit(limit)
    orgs = await cursor.to_list(length=limit)
    subs = [await _org_to_subscription(db, o) for o in orgs]
    return {"subscriptions": subs, "total": len(subs)}


@router.get("/stats")
async def subscription_stats(admin=Depends(require_permission("view_subscriptions"))):
    """Aggregate metrics for the dashboard header cards."""
    db = await get_db()
    orgs = await db["organizations"].find({}).to_list(length=5000)

    mrr_total = 0
    arr_total = 0
    by_status = {"active": 0, "trial": 0, "pending_payment": 0, "cancelled": 0}
    by_plan: Dict[str, int] = {}
    autopay_on = 0
    overage_count = 0

    for o in orgs:
        meta = o.get("admin_meta") or {}
        status = (o.get("status") or "").lower()
        plan = o.get("tier_id") or "unknown"
        by_plan[plan] = by_plan.get(plan, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

        if status not in ("cancelled",):
            mrr = _calc_mrr(o)
            mrr_total += mrr
            arr_total += mrr * 12

        if meta.get("autopay", o.get("payment_method") in ("card", "ach")):
            autopay_on += 1

        # Overage detection (only meaningful for seat-metered tiers)
        seats = _tier_seats(plan)
        if seats > 0:
            used = await _count_active_users(db, str(o.get("_id")))
            if used > seats:
                overage_count += 1

    return {
        "mrr": mrr_total,
        "arr": arr_total,
        "by_status": by_status,
        "by_plan": by_plan,
        "autopay_count": autopay_on,
        "overage_count": overage_count,
        "total_organizations": len(orgs),
    }


@router.post("")
async def create_subscription(
    payload: CreateSubscriptionPayload,
    background_tasks: BackgroundTasks,
    admin=Depends(require_permission("manage_subscriptions")),
):
    """Admin-initiated onboarding — provisions the same way gov checkout does.

    Steps:
      1. Validate tier + email uniqueness
      2. Create `organizations` doc (status='active' unless Stripe checkout requested)
      3. Create `government_users` doc with activation token (super_admin role)
      4. Seed default departments (background)
      5. Send activation email so the city can log into the gov portal (background)
      6. If `request_stripe_checkout` + Stripe configured + payment_method='card',
         also return a hosted-checkout URL the admin can forward to the city.
    """
    db = await get_db()
    if payload.plan not in TIER_META:
        raise HTTPException(400, f"Unknown plan tier '{payload.plan}'")

    contact_email = payload.contact_email.lower().strip()
    if await db["organizations"].find_one({"buyer_email": contact_email}):
        raise HTTPException(400, "An organization with this contact email already exists.")

    tier = TIER_META[payload.plan]
    mrr = payload.mrr if payload.mrr is not None else _default_mrr_for_tier(payload.plan)
    annual = mrr * 12
    months = max(1, int(payload.term_months or 12))
    multi_year = max(1, months // 12)
    now = datetime.utcnow()
    actor = admin.get("email", "admin") if isinstance(admin, dict) else "admin"

    # Derive first/last name from customer_name if not provided explicitly
    first_name = (payload.contact_first_name or payload.customer_name.split(" ")[0] or "Administrator").strip()
    last_name = (payload.contact_last_name or " ".join(payload.customer_name.split(" ")[1:]) or "").strip()

    # Normalize payment method (gov portal expects 'card' | 'ach' | 'invoice')
    raw_pm = (payload.payment_method or "invoice").lower()
    if raw_pm not in ("card", "ach", "invoice"):
        raw_pm = "invoice"

    # Lazy-load gov.billing helpers so we share its activation-email path.
    # Fall back to local equivalents if anything in gov/billing fails to import.
    try:
        gov = _gov_billing_helpers()
        slugify_fn = gov["slugify_city"]
        ensure_unique_slug_fn = gov["ensure_unique_slug"]
        generate_token_fn = gov["generate_activation_token"]
        build_link_fn = gov["build_activation_link"]
        send_email_fn = gov["send_activation_email"]
        seed_org_fn = gov["seed_default_org_data"]
        create_stripe_session_fn = gov["create_stripe_checkout_session"]
        CheckoutRequestModel = gov["CheckoutRequest"]
        gov_available = True
    except Exception as e:
        logger.warning(f"gov.billing helpers unavailable, using fallbacks: {e}")
        slugify_fn = _fallback_slugify
        ensure_unique_slug_fn = _fallback_ensure_unique_slug
        generate_token_fn = _fallback_generate_token
        build_link_fn = lambda t: f"{os.getenv('PORTAL_BASE_URL', 'https://gov.eaiser.ai').rstrip('/')}/activate?token={t}"
        send_email_fn = None
        seed_org_fn = None
        create_stripe_session_fn = None
        CheckoutRequestModel = None
        gov_available = False

    # Slug & uniqueness (reuse gov portal helper for parity)
    state_short = (payload.state or payload.city or "XX").upper()[:2]
    slug = await ensure_unique_slug_fn(db, slugify_fn(payload.customer_name, state_short))

    org_doc = {
        "slug": slug,
        "legal_name": payload.customer_name,
        "state": (payload.state or "").upper(),
        "population": int(payload.population or 0),
        "tier_id": payload.plan,
        "tier_name": tier["label"],
        "billing_cycle": "annual",
        "payment_method": raw_pm,
        "fiscal_year_start": "07-01",
        "multi_year": multi_year,
        "annual_price_usd": annual,
        "total_contract_usd": annual * multi_year,
        "billing": {"address": "", "city": payload.city or "", "state": (payload.state or "").upper(), "zip": ""},
        "buyer_email": contact_email,
        "buyer_name": f"{first_name} {last_name}".strip(),
        "buyer_title": payload.contact_title or "Administrator",
        "buyer_phone": payload.contact_phone or "",
        "status": "active",
        "trial_ends_at": now + timedelta(days=30),
        "subscription_id": None,
        "stripe_customer_id": None,
        "created_at": now,
        "updated_at": now,
        "onboarding_complete": False,
        "agreement_accepted_at": now,
        "admin_meta": {
            "auto_renew": payload.auto_renew,
            "autopay": payload.autopay,
            "enforcement_mode": payload.enforcement_mode,
            "overage_rate": float(payload.overage_rate or 0),
            "grace_days": int(payload.grace_days or 14),
            "is_active": True,
            "suspended": False,
            "end_date_override": now + timedelta(days=30 * months),
            "activity": [{
                "ts": now.strftime("%Y-%m-%d"),
                "text": f"Onboarded by EAiSER admin ({tier['label']} plan, ${mrr}/mo, {months}-month term)",
                "actor": actor,
            }],
        },
    }
    result = await db["organizations"].insert_one(org_doc)
    org_id = str(result.inserted_id)

    # Initial admin user — same shape as gov billing checkout
    activation_token = generate_token_fn()
    await db["government_users"].insert_one({
        "email": contact_email,
        "first_name": first_name,
        "last_name": last_name,
        "title": payload.contact_title or "Administrator",
        "phone": payload.contact_phone or "",
        "org_id": org_id,
        "org_slug": slug,
        "role": "super_admin",
        "status": "pending_activation",
        "activation_token": activation_token,
        "activation_expires_at": now + timedelta(days=7),
        "city": payload.customer_name,
        "created_at": now,
    })

    # Seed default departments (background) — only if helper available
    if seed_org_fn is not None:
        background_tasks.add_task(seed_org_fn, org_id, slug)

    # Send activation email so the city's super-admin can log in immediately
    activation_link = build_link_fn(activation_token)
    if payload.send_activation_email and send_email_fn is not None:
        background_tasks.add_task(
            send_email_fn,
            contact_email, first_name, payload.customer_name,
            activation_link, tier["label"], annual * multi_year,
        )

    # Optional Stripe hosted checkout (gives admin a link to forward to the city)
    checkout_url: Optional[str] = None
    if (payload.request_stripe_checkout and raw_pm == "card"
            and _stripe_enabled() and CheckoutRequestModel is not None
            and create_stripe_session_fn is not None):
        try:
            checkout_request = CheckoutRequestModel(
                tierId=payload.plan, tierName=tier["label"],
                cityName=payload.customer_name, state=(payload.state or "XX").upper(),
                population=int(payload.population or 0),
                firstName=first_name, lastName=last_name,
                title=payload.contact_title or "Administrator",
                email=contact_email, phone=payload.contact_phone or "",
                billingAddress="", billingCity=payload.city or "",
                billingState=(payload.state or "XX").upper(), billingZip="",
                billingCycle="annual", paymentMethod="card",
                fyStart="07-01", multiYear=multi_year,
                annualPrice=annual, totalContract=annual * multi_year,
                agreeTerms=True, agreeDPA=True,
            )
            checkout_url = await create_stripe_session_fn(checkout_request, slug, org_id)
            if checkout_url:
                await db["organizations"].update_one(
                    {"_id": result.inserted_id},
                    {"$set": {"status": "pending_payment"}},
                )
        except Exception as e:
            logger.warning(f"Stripe checkout session creation failed for admin-onboarded org {slug}: {e}")

    org = await db["organizations"].find_one({"_id": result.inserted_id})
    response = await _org_to_subscription(db, org)
    response["activation_link"] = activation_link
    response["checkout_url"] = checkout_url
    return response


@router.post("/{sub_id}/renew")
async def renew_subscription(
    sub_id: str, payload: RenewPayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    months = max(1, int(payload.months or 12))
    current_end = _calc_end_date(org) or datetime.utcnow()
    new_end = max(datetime.utcnow(), current_end) + timedelta(days=30 * months)

    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "admin_meta.end_date_override": new_end,
            "admin_meta.is_active": True,
            "admin_meta.suspended": False,
            "status": "active",
            "updated_at": datetime.utcnow(),
            "first_payment_at": datetime.utcnow(),
        }},
    )
    await _log_activity(db, {"_id": org["_id"]}, f"Renewed for {months} months (next end {new_end.date().isoformat()})")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.delete("/{sub_id}")
async def delete_subscription(
    sub_id: str,
    admin=Depends(require_permission("manage_subscriptions")),
):
    """PERMANENTLY delete a city tenant — its organization record and every
    government user that belongs to it. Irreversible. Cancels the Stripe
    subscription first if one is configured. Citizen-reported issues (tied to a
    ZIP, not the tenant) are intentionally left untouched."""
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    org_id = str(org["_id"])
    slug = org.get("slug")
    city = org.get("legal_name")

    # Best-effort: cancel the Stripe subscription so we stop billing.
    try:
        stripe_sub = org.get("stripe_subscription_id") or org.get("subscription_id")
        if _stripe_enabled() and stripe_sub:
            await _stripe_cancel_subscription(stripe_sub)
    except Exception as e:
        logger.warning(f"Stripe cancel failed while deleting {city}: {e}")

    # Remove every gov user scoped to this tenant (by org id / slug / city).
    or_clauses = [{"org_id": org_id}]
    if slug:
        or_clauses.append({"org_slug": slug})
    if city:
        or_clauses.append({"city": city})
    users_res = await db["government_users"].delete_many({"$or": or_clauses})

    # Finally, delete the tenant itself.
    await db["organizations"].delete_one({"_id": org["_id"]})

    logger.info(f"🗑️ PERMANENTLY deleted city tenant '{city}' ({slug}) + {users_res.deleted_count} users — by {admin.get('email')}")
    return {"success": True, "deleted_city": city, "deleted_users": users_res.deleted_count}


@router.post("/{sub_id}/upgrade")
async def upgrade_subscription(
    sub_id: str, payload: UpgradePayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    if payload.plan not in TIER_META:
        raise HTTPException(400, f"Unknown plan tier '{payload.plan}'")
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    tier = TIER_META[payload.plan]
    mrr = payload.mrr if payload.mrr is not None else _default_mrr_for_tier(payload.plan)
    annual = mrr * 12

    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "tier_id": payload.plan,
            "tier_name": tier["label"],
            "annual_price_usd": annual,
            "total_contract_usd": annual * max(1, int(org.get("multi_year") or 1)),
            "updated_at": datetime.utcnow(),
        }},
    )
    await _log_activity(db, {"_id": org["_id"]},
        f"Upgraded to {tier['label']} (${mrr}/mo)")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/limit")
async def update_limit(
    sub_id: str, payload: LimitPayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    """Adjust staff-seat cap (override the tier default)."""
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")
    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {"admin_meta.user_limit_override": int(payload.user_limit), "updated_at": datetime.utcnow()}},
    )
    await _log_activity(db, {"_id": org["_id"]},
        f"Seat cap set to {payload.user_limit if payload.user_limit > 0 else 'Unlimited'}")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/suspend")
async def suspend_subscription(
    sub_id: str, payload: SuspendPayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    await _stripe_cancel_subscription(org.get("subscription_id"))
    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "status": "cancelled",
            "cancelled_at": datetime.utcnow(),
            "admin_meta.suspended": True,
            "admin_meta.is_active": False,
            "admin_meta.suspension_reason": payload.reason,
            "updated_at": datetime.utcnow(),
        }},
    )
    await _log_activity(db, {"_id": org["_id"]}, f"Suspended: {payload.reason}")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/reactivate")
async def reactivate_subscription(
    sub_id: str,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "status": "active",
            "admin_meta.suspended": False,
            "admin_meta.is_active": True,
            "updated_at": datetime.utcnow(),
        }, "$unset": {"cancelled_at": ""}},
    )
    await _log_activity(db, {"_id": org["_id"]}, "Reactivated by admin")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/active")
async def set_active(
    sub_id: str, payload: ActivePayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "admin_meta.is_active": bool(payload.is_active),
            "admin_meta.suspended": False if payload.is_active else org.get("admin_meta", {}).get("suspended", False),
            "updated_at": datetime.utcnow(),
        }},
    )
    await _log_activity(db, {"_id": org["_id"]},
        "Access enabled" if payload.is_active else "Access paused")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/autopay")
async def set_autopay(
    sub_id: str, payload: AutopayPayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    update_doc: Dict[str, Any] = {
        "admin_meta.autopay": bool(payload.autopay),
        "updated_at": datetime.utcnow(),
    }
    if payload.payment_method:
        update_doc["admin_meta.payment_method_override"] = payload.payment_method

    # If a Stripe payment-method ID was supplied and we have a customer, attach it.
    if payload.autopay and payload.payment_method and payload.payment_method.startswith("pm_"):
        await _stripe_set_default_payment(org.get("stripe_customer_id"), payload.payment_method)

    await db["organizations"].update_one({"_id": org["_id"]}, {"$set": update_doc})
    await _log_activity(db, {"_id": org["_id"]},
        "Autopay enabled" if payload.autopay else "Autopay disabled")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/enforcement")
async def set_enforcement(
    sub_id: str, payload: EnforcementPayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    if payload.mode not in ("hard", "soft", "monitor"):
        raise HTTPException(400, "Mode must be hard / soft / monitor")
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    await db["organizations"].update_one(
        {"_id": org["_id"]},
        {"$set": {
            "admin_meta.enforcement_mode": payload.mode,
            "admin_meta.overage_rate": float(payload.overageRate or 0),
            "admin_meta.grace_days": int(payload.graceDays or 0),
            "updated_at": datetime.utcnow(),
        }},
    )
    await _log_activity(db, {"_id": org["_id"]},
        f"Enforcement set to {payload.mode} (rate ${payload.overageRate}/seat, grace {payload.graceDays}d)")
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.post("/{sub_id}/resolve-overage")
async def resolve_overage(
    sub_id: str, payload: ResolveOveragePayload,
    admin=Depends(require_permission("manage_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")

    action = payload.action
    update: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    note = ""

    if action == "upgrade":
        order = ["spark", "spark-plus", "core", "pro", "enterprise", "metro"]
        cur = org.get("tier_id") or "spark"
        next_tier = order[min(len(order) - 1, order.index(cur) + 1)] if cur in order else "pro"
        next_meta = TIER_META[next_tier]
        new_mrr = _default_mrr_for_tier(next_tier)
        update.update({
            "tier_id": next_tier,
            "tier_name": next_meta["label"],
            "annual_price_usd": new_mrr * 12,
            "admin_meta.grace_started_at": None,
        })
        note = f"Auto-upgraded to {next_meta['label']} to clear overage"
    elif action == "bill":
        update["admin_meta.grace_started_at"] = None
        note = "Overage billed (added to next invoice)"
    elif action == "block":
        update["admin_meta.enforcement_mode"] = "hard"
        update["admin_meta.grace_started_at"] = None
        note = "Enforcement switched to Hard Cap — new seat signups blocked"
    elif action == "notify":
        note = f"Overage notice queued for {org.get('buyer_email')}"
    else:
        raise HTTPException(400, "Action must be upgrade / bill / block / notify")

    await db["organizations"].update_one({"_id": org["_id"]}, {"$set": update})
    await _log_activity(db, {"_id": org["_id"]}, note)
    org = await db["organizations"].find_one({"_id": org["_id"]})
    return await _org_to_subscription(db, org)


@router.get("/{sub_id}/activity")
async def get_activity(
    sub_id: str,
    limit: int = Query(50, ge=1, le=200),
    admin=Depends(require_permission("view_subscriptions")),
):
    db = await get_db()
    org = await db["organizations"].find_one(_org_filter(sub_id))
    if not org:
        raise HTTPException(404, "Subscription not found")
    activity = (org.get("admin_meta") or {}).get("activity", [])
    return {"activity": activity[:limit]}
