"""
Signup endpoint security: IP rate limiting, honeypot detection,
disposable email block, and optional reCAPTCHA verification.

Designed to be backwards-compatible: if Redis or reCAPTCHA are not
configured, falls back gracefully without breaking signups.
"""
import os
import time
import logging
from typing import Optional
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)

# Tunable via environment
SIGNUP_LIMIT_PER_HOUR = int(os.getenv("SIGNUP_LIMIT_PER_HOUR", "5"))
SIGNUP_LIMIT_PER_DAY = int(os.getenv("SIGNUP_LIMIT_PER_DAY", "20"))
RECAPTCHA_SECRET = os.getenv("RECAPTCHA_SECRET_KEY")
REQUIRE_EMAIL_VERIFICATION = os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"

# Common disposable / temp-email domains used by bots
DISPOSABLE_DOMAINS = {
    "necub.com", "mailinator.com", "guerrillamail.com", "10minutemail.com",
    "tempmail.com", "throwaway.email", "trashmail.com", "yopmail.com",
    "fakeinbox.com", "sharklasers.com", "getairmail.com", "dispostable.com",
    "maildrop.cc", "mailnesia.com", "tempmailaddress.com", "emailondeck.com",
    "spam4.me", "tempr.email", "tempemail.com", "tempinbox.com",
    "burnermail.io", "moakt.com", "anonbox.net", "mintemail.com",
}

# In-memory fallback if Redis is unavailable
_memory_buckets: dict[str, list[float]] = {}
_MEM_MAX_KEYS = 5000


def get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For from Render proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


async def _redis_count(redis_service, key: str, window_seconds: int) -> int:
    """Sliding window count via Redis sorted set."""
    try:
        client = getattr(redis_service, "redis", None) or getattr(redis_service, "client", None)
        if client is None:
            return -1
        now = time.time()
        cutoff = now - window_seconds
        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 60)
        results = await pipe.execute()
        return int(results[2])
    except Exception as e:
        logger.warning(f"Redis rate-limit check failed, using memory fallback: {e}")
        return -1


def _memory_count(key: str, window_seconds: int) -> int:
    now = time.time()
    cutoff = now - window_seconds
    bucket = _memory_buckets.get(key, [])
    bucket = [t for t in bucket if t > cutoff]
    bucket.append(now)
    _memory_buckets[key] = bucket
    if len(_memory_buckets) > _MEM_MAX_KEYS:
        # Cheap eviction: drop oldest half
        sorted_keys = sorted(_memory_buckets.items(), key=lambda kv: kv[1][-1] if kv[1] else 0)
        for k, _ in sorted_keys[: _MEM_MAX_KEYS // 2]:
            _memory_buckets.pop(k, None)
    return len(bucket)


async def check_signup_rate_limit(request: Request) -> None:
    """Raise HTTPException 429 if the client IP has exceeded signup limits."""
    ip = get_client_ip(request)
    if ip == "unknown":
        return  # don't penalize unknown when proxy headers missing

    redis_service = None
    try:
        from services.redis_cluster_service import get_redis_cluster_service
        redis_service = await get_redis_cluster_service()
    except Exception:
        redis_service = None

    hour_key = f"signup_rl:hour:{ip}"
    day_key = f"signup_rl:day:{ip}"

    hour_count = -1
    day_count = -1
    if redis_service:
        hour_count = await _redis_count(redis_service, hour_key, 3600)
        day_count = await _redis_count(redis_service, day_key, 86400)

    if hour_count < 0:
        hour_count = _memory_count(hour_key, 3600)
    if day_count < 0:
        day_count = _memory_count(day_key, 86400)

    if hour_count > SIGNUP_LIMIT_PER_HOUR:
        logger.warning(f"[SIGNUP RATE LIMIT] IP={ip} hourly={hour_count}/{SIGNUP_LIMIT_PER_HOUR}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many signup attempts. Please try again later.",
        )
    if day_count > SIGNUP_LIMIT_PER_DAY:
        logger.warning(f"[SIGNUP RATE LIMIT] IP={ip} daily={day_count}/{SIGNUP_LIMIT_PER_DAY}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily signup limit exceeded. Please try again tomorrow.",
        )


def check_honeypot(honeypot_value: Optional[str], request: Request) -> None:
    """Reject bots that fill hidden honeypot field. Real browsers leave it empty."""
    if honeypot_value:
        ip = get_client_ip(request)
        logger.warning(f"[HONEYPOT TRIGGERED] IP={ip} value={honeypot_value[:50]!r}")
        # Return generic success-looking error to confuse bots
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signup data.",
        )


def check_disposable_email(email: str) -> None:
    """Block known disposable / throwaway email domains."""
    if "@" not in email:
        return
    domain = email.split("@", 1)[1].lower().strip()
    if domain in DISPOSABLE_DOMAINS:
        logger.warning(f"[DISPOSABLE EMAIL BLOCKED] {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This email provider is not allowed. Please use a permanent email address.",
        )


async def verify_recaptcha(token: Optional[str], request: Request) -> None:
    """Verify Google reCAPTCHA v3 token. Skipped if RECAPTCHA_SECRET_KEY not set."""
    if not RECAPTCHA_SECRET:
        return  # reCAPTCHA not configured — skip silently
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Captcha verification required.",
        )
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data={
                    "secret": RECAPTCHA_SECRET,
                    "response": token,
                    "remoteip": get_client_ip(request),
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                if not data.get("success") or data.get("score", 0) < 0.5:
                    logger.warning(f"[CAPTCHA FAILED] data={data}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Captcha verification failed.",
                    )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CAPTCHA ERROR] {e}")
        # Don't block signup if Google API is down — fail open
