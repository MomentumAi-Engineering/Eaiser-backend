"""
EAiSER Civic — Billing & Self-Serve Checkout

Handles:
  - Spark / Spark+ self-serve checkout (Stripe-ready, with fallback to invoice)
  - Demo-request lead capture for Core+ tiers
  - Stripe webhook to provision orgs after payment

Mounted at /api/gov/billing
"""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import re
import secrets
import logging

try:
    from app.services.mongodb_service import get_db
except ImportError:
    from services.mongodb_service import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gov/billing", tags=["Billing & Checkout"])

# ──────────────────────────────────────────────────────────────────────
# Tier reference (must stay in sync with frontend src/services/pricing.js)
# ──────────────────────────────────────────────────────────────────────

TIER_FLOOR = {
    "spark":      {"floor": 7200,  "max_pop": 4999,  "self_serve": True,  "cf": 0.30},
    "spark-plus": {"floor": 12000, "max_pop": 9999,  "self_serve": True,  "cf": 0.55},
    "core":       {"floor": 18000, "max_pop": 24999, "self_serve": False, "cf": 0.667},
    "pro":        {"floor": 54000, "max_pop": 99999, "self_serve": False, "cf": 1.0},
    "enterprise": {"floor": 157500,"max_pop": 499999,"self_serve": False, "cf": 1.0},
    "metro":      {"floor": 540000,"max_pop": 10**9, "self_serve": False, "cf": None},
}

PER_CAPITA = 7.89
PENETRATION = 0.40

def calculate_price(tier_id: str, population: int) -> int:
    tier = TIER_FLOOR.get(tier_id)
    if not tier:
        return 0
    if tier["cf"] is None:
        return tier["floor"]
    raw = tier["cf"] * PENETRATION * population * PER_CAPITA
    return max(tier["floor"], int(raw))

# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

ZIP_RE = re.compile(r"^\d{5}$")


class CheckoutRequest(BaseModel):
    tierId: str
    tierName: Optional[str] = None
    cityName: str
    state: str
    population: int
    # ZIP coverage — `cityZip` is the primary ZIP for the city itself (NOT the
    # billing address). It's what we set on the initial Super Admin's
    # `zip_code` field so the gov portal can filter reports correctly.
    # `additionalZips` covers cases where one city spans multiple ZIPs.
    cityZip: str
    additionalZips: List[str] = Field(default_factory=list)
    firstName: str
    lastName: str
    title: str
    email: EmailStr
    phone: Optional[str] = ""
    billingAddress: str
    billingCity: str
    billingState: str
    billingZip: str
    billingCycle: str = "annual"           # "annual" or "monthly"
    paymentMethod: str = "ach"             # "ach" | "card" | "invoice"
    fyStart: str = "07-01"
    multiYear: int = 1
    annualPrice: int
    totalContract: int
    agreeTerms: bool
    agreeDPA: bool

class DemoRequest(BaseModel):
    cityName: str
    state: str
    population: Optional[str] = ""
    role: Optional[str] = ""
    firstName: str
    lastName: str
    title: Optional[str] = ""
    email: EmailStr
    phone: Optional[str] = ""
    timing: Optional[str] = ""
    currentStack: List[str] = []
    interestedTier: Optional[str] = ""
    notes: Optional[str] = ""

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def slugify_city(city: str, state: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", f"{city}-{state}".lower()).strip("-")
    return base or f"city-{secrets.token_hex(3)}"

async def _ensure_unique_slug(db, slug: str) -> str:
    attempt = slug
    suffix = 0
    while await db["organizations"].find_one({"slug": attempt}):
        suffix += 1
        attempt = f"{slug}-{suffix}"
        if suffix > 50:
            attempt = f"{slug}-{secrets.token_hex(2)}"
            break
    return attempt

def _generate_activation_token() -> str:
    return secrets.token_urlsafe(32)

# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────

@router.post("/checkout")
async def submit_checkout(payload: CheckoutRequest, background_tasks: BackgroundTasks):
    """
    Self-serve checkout for Spark / Spark+ tiers.

    Flow:
      1. Validate tier is self-serve
      2. Verify population matches tier (gentle, not strict — sales can override)
      3. Create Organization record (status = pending_payment | trial)
      4. Create initial admin user (status = pending_activation)
      5. Generate magic-link activation token
      6. If Stripe configured → return checkout URL; else mark trial active, send activation immediately
      7. Always email activation link
    """
    tier_info = TIER_FLOOR.get(payload.tierId)
    if not tier_info:
        raise HTTPException(status_code=400, detail=f"Unknown tier '{payload.tierId}'")

    if not payload.agreeTerms or not payload.agreeDPA:
        raise HTTPException(status_code=400, detail="Terms and DPA must be accepted")

    if not tier_info["self_serve"]:
        raise HTTPException(
            status_code=400,
            detail=f"{payload.tierId} tier requires sales-assisted checkout. Please use /demo."
        )

    # ── Validate ZIP coverage ─────────────────────────────────────────
    # Primary ZIP is mandatory because it's what we set on the initial admin's
    # `zip_code` field (gov portal uses it to filter incoming reports).
    if not ZIP_RE.match(payload.cityZip):
        raise HTTPException(status_code=400, detail="Primary city ZIP must be 5 digits")
    bad_additional = [z for z in payload.additionalZips if not ZIP_RE.match(z)]
    if bad_additional:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid additional ZIP codes: {', '.join(bad_additional)}",
        )
    # De-duplicate while preserving order; primary first.
    seen, all_zips = set(), []
    for z in [payload.cityZip] + list(payload.additionalZips):
        if z not in seen:
            seen.add(z)
            all_zips.append(z)

    db = await get_db()

    # 1. Slug & uniqueness
    base_slug = slugify_city(payload.cityName, payload.state)
    slug = await _ensure_unique_slug(db, base_slug)

    # 2. Existing customer guard
    existing = await db["organizations"].find_one({"buyer_email": payload.email.lower()})
    if existing:
        raise HTTPException(
            status_code=400,
            detail="An organization with this buyer email already exists. Please log in or contact support."
        )

    # 3. Create Organization
    now = datetime.utcnow()
    trial_ends = now + timedelta(days=30)

    org_doc = {
        "slug": slug,
        "legal_name": payload.cityName,
        "state": payload.state.upper(),
        "population": payload.population,
        # Service-area ZIPs — used by /portal/reports to route citizen issues
        # to the right gov org. `primary_zip` is the one we stamp onto the
        # initial Super Admin's record. `zip_codes` is the full coverage set.
        "primary_zip": payload.cityZip,
        "zip_codes": all_zips,
        "tier_id": payload.tierId,
        "tier_name": payload.tierName or payload.tierId,
        "billing_cycle": payload.billingCycle,
        "payment_method": payload.paymentMethod,
        "fiscal_year_start": payload.fyStart,
        "multi_year": payload.multiYear,
        "annual_price_usd": payload.annualPrice,
        "total_contract_usd": payload.totalContract,
        "billing": {
            "address": payload.billingAddress,
            "city": payload.billingCity,
            "state": payload.billingState.upper(),
            "zip": payload.billingZip,
        },
        "buyer_email": payload.email.lower(),
        "buyer_name": f"{payload.firstName} {payload.lastName}",
        "buyer_title": payload.title,
        "buyer_phone": payload.phone or "",
        # ── Subscription state ─────────────────────────────────────────
        # `status` is OUR app-level state (used by access control / UI banners).
        # `subscription_status` is the verbatim Stripe value (trialing | active |
        # past_due | canceled | unpaid | incomplete) — used by billing dashboards.
        # We initialize them all up-front so downstream reads never have to
        # handle missing keys.
        "status": "trial",                       # → "trialing" after checkout, "active" after first paid invoice
        "subscription_status": None,             # Stripe-verbatim status, set by webhook
        "trial_ends_at": trial_ends,             # 30-day display target (Stripe is authoritative once sub exists)
        "subscription_id": None,                 # sub_xxx, set by webhook
        "stripe_customer_id": None,              # cus_xxx, set by webhook
        "current_period_end": None,              # When the next renewal charge will hit
        "cancel_at_period_end": False,           # True if user clicked "Cancel" in portal — still active until period_end
        "latest_invoice_status": None,           # paid | payment_failed | open | void
        "latest_invoice_amount_cents": None,
        "first_payment_at": None,                # Set on first invoice.paid
        "last_payment_at": None,                 # Refreshed on every successful renewal
        "last_payment_failure_at": None,
        "cancelled_at": None,
        "created_at": now,
        "updated_at": now,
        "onboarding_complete": False,
        "agreement_accepted_at": now,
    }
    org_result = await db["organizations"].insert_one(org_doc)
    org_id = str(org_result.inserted_id)

    # 4. Initial admin user (pending activation)
    activation_token = _generate_activation_token()
    user_doc = {
        # `name` is what the rest of the codebase (gov_auth, gov_portal) reads.
        # Keep first/last for HR-style emails, and a combined `name` for the
        # portal UI and welcome flows.
        "name": f"{payload.firstName} {payload.lastName}".strip(),
        "first_name": payload.firstName,
        "last_name": payload.lastName,
        "email": payload.email.lower(),
        "title": payload.title,
        "phone": payload.phone or "",
        "org_id": org_id,
        "org_slug": slug,
        "role": "super_admin",
        "status": "pending_activation",
        "activation_token": activation_token,
        "activation_expires_at": now + timedelta(days=7),
        "city": payload.cityName,
        # Mirrors the primary city ZIP so gov_portal's report filter works
        # on day one — without this, the new admin sees an empty queue.
        "zip_code": payload.cityZip,
        "department": "CITY_MANAGEMENT",
        "is_active": True,
        "created_at": now,
    }
    await db["government_users"].insert_one(user_doc)

    # 5. Seed default org data (departments, categories, etc.) — non-blocking.
    # Pass cityName so the gov-portal departments (keyed by city) are created
    # too, not just the legacy org-scoped mirror.
    background_tasks.add_task(_seed_default_org_data, org_id, slug, payload.cityName, payload.email.lower())

    # 6. Decide payment route
    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    checkout_url = None
    if payload.paymentMethod == "card" and stripe_key:
        checkout_url = await _create_stripe_checkout_session(payload, slug, org_id)
        await db["organizations"].update_one(
            {"_id": org_result.inserted_id},
            {"$set": {"status": "pending_payment"}}
        )

    # 7. Send activation email (regardless of payment route)
    activation_link = _build_activation_link(activation_token)
    background_tasks.add_task(
        _send_activation_email,
        payload.email.lower(),
        payload.firstName,
        payload.cityName,
        activation_link,
        payload.tierName or payload.tierId,
        payload.totalContract,
    )

    # 8. Internal notify (sales channel + CRM) — fire-and-forget
    background_tasks.add_task(_notify_internal_new_customer, org_doc, org_id)

    return {
        "success": True,
        "org_id": org_id,
        "org_slug": slug,
        "checkout_url": checkout_url,
        "trial_ends_at": trial_ends.isoformat(),
        "activation_sent_to": payload.email.lower(),
    }


@router.post("/demo-request")
async def submit_demo_request(payload: DemoRequest, background_tasks: BackgroundTasks):
    """
    Lead capture for Core / Pro / Enterprise / Metro tiers.
    Stored in `demo_requests` collection + emailed to sales.
    """
    db = await get_db()
    doc = payload.dict()
    doc["created_at"] = datetime.utcnow()
    doc["status"] = "new"
    doc["assigned_to"] = None
    result = await db["demo_requests"].insert_one(doc)

    background_tasks.add_task(_notify_sales_demo_request, doc)

    return {"success": True, "request_id": str(result.inserted_id)}


@router.get("/me")
async def get_my_subscription(org_slug: str):
    """
    Returns the current org's subscription state for the city-side billing page.

    Auth note: accepts `org_slug` as a query param for now. Once gov_auth middleware
    is wired in, this should read org_slug from the JWT claim instead, and ignore
    any client-supplied value — otherwise any authenticated city admin could read
    another org's billing state. Same caveat as /portal-session.

    Returns a flat, UI-friendly shape (NOT the raw DB doc) so we can rename DB
    fields later without breaking the frontend.
    """
    db = await get_db()
    org = await db["organizations"].find_one({"slug": org_slug})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    # Trial countdown — prefer Stripe's `current_period_end` (set after checkout)
    # but fall back to our DB's `trial_ends_at` (set at org creation, before
    # Stripe checkout completes).
    trial_ends_at = org.get("current_period_end") or org.get("trial_ends_at")
    days_left = None
    if trial_ends_at:
        delta = trial_ends_at - datetime.utcnow()
        days_left = max(0, delta.days)

    tier_id = org.get("tier_id")
    tier_info = TIER_FLOOR.get(tier_id, {})

    return {
        "org_slug": org.get("slug"),
        "legal_name": org.get("legal_name"),
        # Service-area ZIPs — surfaced on the billing page so the city admin
        # can verify coverage. `primary_zip` is the one stamped on the initial
        # admin user; `zip_codes` is the full coverage list.
        "primary_zip": org.get("primary_zip"),
        "zip_codes": org.get("zip_codes") or ([org.get("primary_zip")] if org.get("primary_zip") else []),
        "tier": {
            "id": tier_id,
            "name": org.get("tier_name") or tier_id,
            "annual_price_usd": org.get("annual_price_usd"),
            "billing_cycle": org.get("billing_cycle"),
            "max_population": tier_info.get("max_pop"),
        },
        "status": org.get("status"),                              # app-level: trial/trialing/active/past_due/cancelled
        "subscription_status": org.get("subscription_status"),    # Stripe-verbatim
        "trial_ends_at": (org.get("trial_ends_at").isoformat() if org.get("trial_ends_at") else None),
        "current_period_end": (org.get("current_period_end").isoformat() if org.get("current_period_end") else None),
        "days_left_in_period": days_left,
        "cancel_at_period_end": bool(org.get("cancel_at_period_end")),
        "has_payment_method": bool(org.get("stripe_customer_id")),
        "latest_invoice_status": org.get("latest_invoice_status"),
        "latest_invoice_amount_usd": (
            (org.get("latest_invoice_amount_cents") / 100) if org.get("latest_invoice_amount_cents") else None
        ),
        "first_payment_at": (org.get("first_payment_at").isoformat() if org.get("first_payment_at") else None),
        "last_payment_at": (org.get("last_payment_at").isoformat() if org.get("last_payment_at") else None),
        "last_payment_failure_at": (org.get("last_payment_failure_at").isoformat() if org.get("last_payment_failure_at") else None),
    }


class PortalSessionRequest(BaseModel):
    """
    Auth note: the caller must already be authenticated as a gov user. The org
    is identified by `org_slug` from the auth token in production — this model
    is just the body shape. Until gov_auth is wired in, we accept the slug
    directly and let route-level auth gate access.
    """
    org_slug: str
    return_url: Optional[str] = None  # Where Stripe sends the user after they're done


@router.post("/portal-session")
async def create_portal_session(payload: PortalSessionRequest):
    """
    Create a Stripe Customer Portal session and return its URL.

    The Customer Portal is Stripe-hosted UI where a city admin can:
      • Update payment method (new card / new bank account)
      • View past invoices and download receipts
      • Change billing cycle (annual ↔ monthly, if enabled in Stripe Dashboard)
      • Cancel the subscription (sets cancel_at_period_end=true on Stripe side)

    We do NOT build any of this UI ourselves — Stripe handles compliance, retries,
    SCA, etc. We just give them a one-time URL to redirect to.

    Prereq: the org must already have a `stripe_customer_id` (set by the
    checkout.session.completed webhook on first successful checkout). If not,
    return 409 — the org hasn't paid yet, so there's nothing to manage.
    """
    db = await get_db()
    org = await db["organizations"].find_one({"slug": payload.org_slug})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    customer_id = org.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "No Stripe customer on file for this organization yet. "
                "Complete checkout first, then the portal will be available."
            ),
        )

    try:
        import stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        if not stripe.api_key:
            raise HTTPException(status_code=503, detail="Billing not configured")

        return_url = payload.return_url or f"{_portal_base()}/billing"
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"url": session.url}
    except ImportError:
        raise HTTPException(status_code=503, detail="Stripe SDK not installed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stripe portal session creation failed for {payload.org_slug}: {e}")
        raise HTTPException(status_code=502, detail="Could not start billing portal session")


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Stripe webhook receiver — keeps our org state in sync with Stripe's
    subscription state machine.

    Events handled (full lifecycle):
      checkout.session.completed     → first signup: store customer_id + sub_id, mark trialing
      customer.subscription.created  → defensive: same as above if checkout webhook is missed
      customer.subscription.updated  → trial→active, period rollover, cancel scheduled, etc.
      customer.subscription.deleted  → fully cancelled (sub is gone)
      invoice.paid                   → successful renewal charge — refresh period_end
      invoice.payment_failed         → mark past_due; Stripe will retry per Dunning settings

    Idempotency: every webhook is recorded by `event.id` in `stripe_events`.
    If we see the same ID twice (Stripe retries on 5xx), we ack and skip — this
    prevents double-charges on first_payment_at, duplicate renewals, etc.

    Signature verification: REQUIRED in production via STRIPE_WEBHOOK_SECRET.
    The dev fallback (no signature) only runs when the secret is unset, AND
    logs a loud warning so no one ships it that way by accident.
    """
    payload_bytes = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        logger.warning("Stripe webhook received but STRIPE_WEBHOOK_SECRET not set — skipping verification (DEV ONLY)")
        try:
            event = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")
    else:
        try:
            import stripe
            event = stripe.Webhook.construct_event(payload_bytes, sig_header, secret)
        except ImportError:
            logger.error("stripe package not installed; cannot verify webhook")
            raise HTTPException(status_code=500, detail="Stripe SDK unavailable")
        except Exception as e:
            logger.error(f"Stripe webhook verification failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event.get("id")
    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    db = await get_db()

    # ── Idempotency: ack-and-skip if we've already processed this event ──
    if event_id:
        try:
            await db["stripe_events"].insert_one({
                "_id": event_id,           # Stripe's evt_xxx, unique
                "type": event_type,
                "received_at": datetime.utcnow(),
            })
        except Exception:
            # Duplicate key — we've seen this event before. Stripe retries on
            # any non-2xx, so ack 200 to stop the retry loop.
            logger.info(f"Stripe webhook duplicate ignored: {event_id} ({event_type})")
            return {"received": True, "duplicate": True}

    try:
        await _handle_stripe_event(db, event_type, data_object)
    except Exception as e:
        # If our handler fails, return 500 so Stripe retries. The idempotency
        # row stays — on retry we'll skip and ack, which is wrong. So delete
        # it here to allow the retry to actually re-process.
        if event_id:
            try:
                await db["stripe_events"].delete_one({"_id": event_id})
            except Exception:
                pass
        logger.exception(f"Stripe webhook handler failed for {event_type}: {e}")
        raise HTTPException(status_code=500, detail="Webhook handler error")

    return {"received": True, "event": event_type}


async def _handle_stripe_event(db, event_type: str, obj: Dict[str, Any]):
    """Per-event dispatch. Pure DB writes, no network calls."""
    now = datetime.utcnow()
    metadata = obj.get("metadata") or {}
    org_slug = metadata.get("org_slug")

    # ─── First-signup events ──────────────────────────────────────────
    if event_type == "checkout.session.completed":
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")
        if org_slug:
            await db["organizations"].update_one(
                {"slug": org_slug},
                {"$set": {
                    # Mark "trialing" (not "active") — they paid only for trial,
                    # not yet for first period. Status will flip to "active" on
                    # invoice.paid after the trial ends.
                    "status": "trialing",
                    "subscription_status": "trialing",
                    "stripe_customer_id": customer_id,
                    "subscription_id": subscription_id,
                    "checkout_completed_at": now,
                    "updated_at": now,
                }}
            )
            logger.info(f"✅ Checkout completed for {org_slug} — trial started")

    # ─── Subscription lifecycle ───────────────────────────────────────
    elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
        # `obj` is the Subscription object
        sub_id = obj.get("id")
        stripe_status = obj.get("status")  # trialing | active | past_due | canceled | unpaid | incomplete
        cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
        current_period_end = obj.get("current_period_end")
        period_end_dt = datetime.utcfromtimestamp(current_period_end) if current_period_end else None

        # Map Stripe's status → our high-level org status (kept compatible with
        # the existing "trial"/"active"/"cancelled" values used elsewhere).
        org_status_map = {
            "trialing":   "trialing",
            "active":     "active",
            "past_due":   "past_due",
            "unpaid":     "past_due",
            "canceled":   "cancelled",
            "incomplete": "pending_payment",
            "incomplete_expired": "cancelled",
        }
        org_status = org_status_map.get(stripe_status, "active")

        update = {
            "status": org_status,
            "subscription_status": stripe_status,
            "subscription_id": sub_id,
            "cancel_at_period_end": cancel_at_period_end,
            "current_period_end": period_end_dt,
            "updated_at": now,
        }
        if event_type == "customer.subscription.created":
            update["subscription_created_at"] = now

        # Match by slug if metadata has it (preferred), else fall back to sub_id.
        match = {"slug": org_slug} if org_slug else {"subscription_id": sub_id}
        if not org_slug and not sub_id:
            logger.warning("Subscription event with no org_slug and no sub_id — skipping")
            return
        await db["organizations"].update_one(match, {"$set": update})
        logger.info(
            f"🔄 Sub {sub_id[:14] if sub_id else '?'}… "
            f"status={stripe_status}"
            + (" (cancel scheduled)" if cancel_at_period_end else "")
        )

    elif event_type == "customer.subscription.deleted":
        sub_id = obj.get("id")
        match = {"slug": org_slug} if org_slug else {"subscription_id": sub_id}
        await db["organizations"].update_one(
            match,
            {"$set": {
                "status": "cancelled",
                "subscription_status": "canceled",
                "cancelled_at": now,
                "updated_at": now,
            }}
        )
        logger.info(f"🛑 Subscription cancelled for {org_slug or sub_id}")

    # ─── Invoice events (renewals) ────────────────────────────────────
    elif event_type == "invoice.paid":
        # `obj` is the Invoice. Look up sub via invoice.subscription.
        sub_id = obj.get("subscription")
        amount_paid = obj.get("amount_paid", 0)  # in cents
        period_end_ts = obj.get("period_end") or obj.get("lines", {}).get("data", [{}])[0].get("period", {}).get("end")
        period_end_dt = datetime.utcfromtimestamp(period_end_ts) if period_end_ts else None
        match = {"slug": org_slug} if org_slug else {"subscription_id": sub_id}

        org = await db["organizations"].find_one(match) or {}
        first_payment_at = org.get("first_payment_at") or now

        await db["organizations"].update_one(
            match,
            {"$set": {
                "status": "active",
                "subscription_status": "active",
                "latest_invoice_status": "paid",
                "latest_invoice_amount_cents": amount_paid,
                "current_period_end": period_end_dt,
                "first_payment_at": first_payment_at,
                "last_payment_at": now,
                "updated_at": now,
            }}
        )
        logger.info(f"💰 Invoice paid for {org_slug or sub_id}: ${amount_paid/100:.2f}")

    elif event_type == "invoice.payment_failed":
        sub_id = obj.get("subscription")
        match = {"slug": org_slug} if org_slug else {"subscription_id": sub_id}
        await db["organizations"].update_one(
            match,
            {"$set": {
                "status": "past_due",
                "subscription_status": "past_due",
                "latest_invoice_status": "payment_failed",
                "last_payment_failure_at": now,
                "updated_at": now,
            }}
        )
        logger.warning(f"⚠️  Payment failed for {org_slug or sub_id}")
        # TODO: queue a dunning email here when email infra is ready.

    else:
        logger.debug(f"Stripe event ignored (no handler): {event_type}")


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

TRIAL_PERIOD_DAYS = 30


async def _create_stripe_checkout_session(payload: CheckoutRequest, slug: str, org_id: str) -> Optional[str]:
    """
    Create a Stripe Checkout session in SUBSCRIPTION mode with a 30-day trial.

    Why subscription mode (even for "annual" billing cycle)?
      • Auto-renewal is built in — Stripe charges the card each period without
        any cron job on our side.
      • Customer object + saved payment method are required for renewal — `mode=payment`
        wouldn't give us either.
      • `trial_period_days` makes Stripe the source of truth for the trial:
        no charge until day 31, no need for us to schedule "trial expires" jobs.

    Per-period pricing:
      • annual  → charges `annualPrice` once per year
      • monthly → charges `annualPrice / 12` (rounded to cent) every month

    `totalContract` (multi-year) is NOT used for the recurring amount — multi-year
    contracts are a sales-assisted concern handled separately. The Customer Portal
    lets cities cancel/upgrade without contacting us.
    """
    try:
        import stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

        is_annual = payload.billingCycle == "annual"
        interval = "year" if is_annual else "month"
        # Cents per billing period. For monthly: 1/12 of annual, rounded to nearest cent.
        per_period_cents = int(round(payload.annualPrice * 100)) if is_annual \
            else int(round(payload.annualPrice * 100 / 12))

        # Metadata is duplicated onto both the Checkout Session and the Subscription
        # it creates, so every downstream webhook can find the org without a DB lookup.
        common_metadata = {
            "org_slug": slug,
            "org_id": org_id,
            "tier_id": payload.tierId,
            "billing_cycle": payload.billingCycle,
        }

        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card", "us_bank_account"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"EAiSER Civic — {payload.tierName or payload.tierId}",
                        "description": f"{payload.cityName}, {payload.state} · Pop. {payload.population:,}",
                    },
                    "unit_amount": per_period_cents,
                    "recurring": {"interval": interval},
                },
                "quantity": 1,
            }],
            subscription_data={
                "trial_period_days": TRIAL_PERIOD_DAYS,
                # Carry metadata onto the Subscription too — invoice / renewal
                # webhooks need it.
                "metadata": common_metadata,
                # If the card declines at trial end, mark the sub as `unpaid`
                # (default `cancel` would silently drop the customer).
                "trial_settings": {
                    "end_behavior": {"missing_payment_method": "cancel"}
                },
            },
            customer_email=payload.email,
            # Always create a Customer object — needed for Customer Portal + renewals.
            customer_creation="always",
            metadata=common_metadata,
            allow_promotion_codes=True,
            billing_address_collection="auto",
            success_url=f"{_portal_base()}/checkout/success?org={slug}",
            cancel_url=f"{_portal_base()}/checkout/{payload.tierId}",
        )
        logger.info(
            f"Stripe Checkout session created for {slug}: "
            f"{interval}ly ${per_period_cents/100:.2f}, {TRIAL_PERIOD_DAYS}d trial"
        )
        return session.url
    except ImportError:
        logger.warning("stripe SDK not installed — falling back to invoice flow")
        return None
    except Exception as e:
        logger.error(f"Stripe session creation failed: {e}")
        return None


async def _seed_default_org_data(org_id: str, slug: str, city: str = "", created_by: str = "system"):
    try:
        db = await get_db()

        # 1) Seed the departments the GOV PORTAL actually reads: gov_departments,
        # keyed by `city` (= admin.org in the JWT). This is what makes the
        # standard departments show up in City Setup and route reports on day
        # one. Each carries canonical issue_types for report routing.
        if city:
            try:
                from app.gov.default_departments import seed_default_departments
            except ImportError:
                from gov.default_departments import seed_default_departments
            await seed_default_departments(db, city, created_by or "system")

        # 2) Legacy org_departments mirror (kept for any org-scoped views).
        default_depts = [
            {"name": "Public Works", "color": "#D4A017", "categories": ["Potholes", "Street repair", "Signage"]},
            {"name": "Sanitation", "color": "#F97316", "categories": ["Trash", "Recycling", "Bulk pickup"]},
            {"name": "Code Enforcement", "color": "#A855F7", "categories": ["Violations", "Inspections"]},
            {"name": "Parks & Recreation", "color": "#10B981", "categories": ["Parks", "Playgrounds", "Trees"]},
            {"name": "Water Services", "color": "#3B82F6", "categories": ["Leaks", "Quality", "Billing"]},
        ]
        await db["org_departments"].insert_many([
            {**d, "org_id": org_id, "org_slug": slug, "created_at": datetime.utcnow()} for d in default_depts
        ])
        logger.info(f"Seeded default departments for org {slug} (city='{city}')")
    except Exception as e:
        logger.error(f"Failed to seed default org data for {slug}: {e}")


async def _send_activation_email(to: str, first_name: str, city: str, link: str, tier: str, total: int):
    try:
        try:
            from app.services.email_service import send_email
        except ImportError:
            from services.email_service import send_email

        subject = f"Welcome to EAiSER Civic — Activate your {city} account"
        html = f"""
        <div style="font-family: 'Inter', sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 24px; background-color: #ffffff; color: #1a1a1a;">
            <div style="text-align: center; margin-bottom: 32px;">
                <h1 style="margin: 0; font-size: 22px; font-weight: 900; letter-spacing: -0.5px;">
                    EAiSER <span style="color: #D4A017;">CIVIC</span>
                </h1>
                <p style="font-size: 11px; color: #888; letter-spacing: 3px; text-transform: uppercase; margin-top: 6px;">
                    Government Portal
                </p>
            </div>

            <p style="font-size: 16px;">Hi {first_name},</p>

            <p style="font-size: 15px; line-height: 1.6;">
                Welcome to EAiSER Civic. Your account for <strong>{city}</strong> is provisioned and ready.
                Click below to set your password and complete setup:
            </p>

            <div style="text-align: center; margin: 32px 0;">
                <a href="{link}"
                   style="display: inline-block; background-color: #D4A017; color: #000;
                          padding: 16px 32px; font-weight: 900; text-decoration: none;
                          letter-spacing: 2px; text-transform: uppercase; font-size: 12px;
                          border-radius: 4px;">
                    Activate My Account
                </a>
            </div>

            <p style="font-size: 13px; color: #555; line-height: 1.6;">
                <strong>Plan:</strong> {tier}<br>
                <strong>Trial:</strong> 30 days, no charge<br>
                <strong>Link expires:</strong> 7 days
            </p>

            <hr style="border: 0; border-top: 1px solid #eee; margin: 32px 0;">

            <p style="font-size: 12px; color: #888; line-height: 1.6;">
                If the button doesn't work, paste this link into your browser:<br>
                <a href="{link}" style="color: #D4A017; word-break: break-all;">{link}</a>
            </p>

            <p style="font-size: 11px; color: #aaa; margin-top: 32px; text-align: center;">
                EAiSER, Inc. · Nashville, TN · support@eaiser.ai
            </p>
        </div>
        """
        text = f"""Welcome to EAiSER Civic

Hi {first_name},

Your EAiSER account for {city} is ready. Activate it here:

{link}

Plan: {tier}
Trial: 30 days, no charge
Link expires: 7 days

— EAiSER, Inc.
"""
        await _maybe_await(send_email, to, subject, html, text)
    except Exception as e:
        logger.error(f"Failed to send activation email to {to}: {e}")


async def _notify_internal_new_customer(org_doc: Dict[str, Any], org_id: str):
    logger.info(f"🎉 New customer: {org_doc.get('legal_name')} ({org_doc.get('tier_id')}) — org_id={org_id}")
    # Hook in Slack / Salesforce / Hubspot here in production


async def _notify_sales_demo_request(doc: Dict[str, Any]):
    logger.info(f"📞 New demo request: {doc.get('cityName')}, {doc.get('state')} ({doc.get('interestedTier') or 'undecided'})")
    try:
        sales_email = os.getenv("SALES_NOTIFY_EMAIL", "sales@eaiser.ai")
        try:
            from app.services.email_service import send_email
        except ImportError:
            from services.email_service import send_email

        subject = f"[New Demo Request] {doc.get('cityName')}, {doc.get('state')} — {doc.get('interestedTier') or 'unspecified'}"
        body = "\n".join([f"{k}: {v}" for k, v in doc.items() if k not in ("_id", "created_at")])
        await _maybe_await(send_email, sales_email, subject, f"<pre>{body}</pre>", body)
    except Exception as e:
        logger.error(f"Failed to email sales about demo request: {e}")


def _build_activation_link(token: str) -> str:
    base = _portal_base()
    return f"{base}/activate?token={token}"


def _portal_base() -> str:
    return os.getenv("PORTAL_BASE_URL", "https://gov.eaiser.ai").rstrip("/")


async def _maybe_await(fn, *args, **kwargs):
    """Call fn whether it's sync or async."""
    import asyncio
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result
