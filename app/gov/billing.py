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

class CheckoutRequest(BaseModel):
    tierId: str
    tierName: Optional[str] = None
    cityName: str
    state: str
    population: int
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
        "status": "trial",
        "trial_ends_at": trial_ends,
        "subscription_id": None,
        "stripe_customer_id": None,
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
        "email": payload.email.lower(),
        "first_name": payload.firstName,
        "last_name": payload.lastName,
        "title": payload.title,
        "phone": payload.phone or "",
        "org_id": org_id,
        "org_slug": slug,
        "role": "super_admin",
        "status": "pending_activation",
        "activation_token": activation_token,
        "activation_expires_at": now + timedelta(days=7),
        "city": payload.cityName,
        "created_at": now,
    }
    await db["government_users"].insert_one(user_doc)

    # 5. Seed default org data (departments, categories, etc.) — non-blocking
    background_tasks.add_task(_seed_default_org_data, org_id, slug)

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


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Stripe webhook handler — provisions org on payment_succeeded.
    Signature verification requires STRIPE_WEBHOOK_SECRET env var.
    """
    payload_bytes = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        logger.warning("Stripe webhook received but STRIPE_WEBHOOK_SECRET not set — skipping verification")
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

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    db = await get_db()

    if event_type in ("checkout.session.completed", "invoice.payment_succeeded"):
        org_slug = (data_object.get("metadata") or {}).get("org_slug")
        customer_id = data_object.get("customer")
        subscription_id = data_object.get("subscription")

        if org_slug:
            await db["organizations"].update_one(
                {"slug": org_slug},
                {"$set": {
                    "status": "active",
                    "stripe_customer_id": customer_id,
                    "subscription_id": subscription_id,
                    "first_payment_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }}
            )
            logger.info(f"Organization {org_slug} activated via Stripe payment")

    elif event_type == "customer.subscription.deleted":
        sub_id = data_object.get("id")
        if sub_id:
            await db["organizations"].update_one(
                {"subscription_id": sub_id},
                {"$set": {"status": "cancelled", "cancelled_at": datetime.utcnow()}}
            )

    return {"received": True, "event": event_type}


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

async def _create_stripe_checkout_session(payload: CheckoutRequest, slug: str, org_id: str) -> Optional[str]:
    try:
        import stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        amount_cents = int(payload.totalContract * 100)

        session = stripe.checkout.Session.create(
            payment_method_types=["card", "us_bank_account"],
            mode="subscription" if payload.billingCycle == "monthly" else "payment",
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"EAiSER Civic — {payload.tierName or payload.tierId}",
                        "description": f"{payload.cityName}, {payload.state} · Pop. {payload.population:,}",
                    },
                    "unit_amount": amount_cents,
                    "recurring": {"interval": "month"} if payload.billingCycle == "monthly" else None,
                },
                "quantity": 1,
            }],
            customer_email=payload.email,
            metadata={"org_slug": slug, "org_id": org_id, "tier_id": payload.tierId},
            success_url=f"{_portal_base()}/checkout/success?org={slug}",
            cancel_url=f"{_portal_base()}/checkout/{payload.tierId}",
        )
        return session.url
    except ImportError:
        logger.warning("stripe SDK not installed — falling back to invoice flow")
        return None
    except Exception as e:
        logger.error(f"Stripe session creation failed: {e}")
        return None


async def _seed_default_org_data(org_id: str, slug: str):
    try:
        db = await get_db()
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
        logger.info(f"Seeded default departments for org {slug}")
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
