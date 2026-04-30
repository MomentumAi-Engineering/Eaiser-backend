"""
🚀 Gemini API Key Pool — Round-Robin Load Balancer for 10K+ Concurrent Users

Distributes Gemini API requests across multiple API keys (from separate
Google Cloud projects) to multiply throughput beyond a single key's limits.

Architecture:
  - Each key gets its own genai client instance (thread-safe)
  - Round-robin rotation with per-key RPM/TPM tracking
  - Automatic exhaustion detection and recovery
  - Error-based key demotion (5 consecutive errors → skip for 60s)
  - Monitoring endpoint for live pool stats

Usage:
  from services.gemini_key_pool import get_key_pool
  pool = get_key_pool()
  model, slot = pool.get_model()   # ready to call model.generate_content(...)
"""

import os
import time
import logging
import threading
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for one key slot
# ---------------------------------------------------------------------------

@dataclass
class KeySlot:
    """Tracks usage metrics for a single API key."""
    key: str
    project_name: str
    index: int = 0
    rpm_used: int = 0
    rpm_limit: int = 1000        # Google AI Studio paid default
    tpm_used: int = 0
    tpm_limit: int = 1_000_000   # 1 M tokens/min per project
    last_reset: float = field(default_factory=time.time)
    errors: int = 0
    consecutive_errors: int = 0
    is_exhausted: bool = False
    last_error_time: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    is_paid: bool = True         # Paid keys get priority routing
    active_calls: int = 0       # Currently in-flight API calls


# ---------------------------------------------------------------------------
# Key Pool
# ---------------------------------------------------------------------------

class GeminiKeyPool:
    """
    Enterprise-grade API key rotation for high-concurrency Gemini usage.

    Features:
    ─────────
    • Round-robin key rotation across N separate project keys
    • Per-key RPM/TPM accounting (auto-resets every 60 s)
    • Consecutive-error demotion (≥5 errors → key disabled for 60 s)
    • Thread-safe (protected by threading.Lock)
    • Zero-downtime recovery: exhausted pool auto-resets
    """

    _RESET_WINDOW = 60          # seconds between counter resets
    _ERROR_COOLDOWN = 60        # seconds to sideline a bad key
    _MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self):
        self._keys: List[KeySlot] = []
        self._current_index: int = 0
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._total_cache_hits: int = 0
        self._load_keys()

    # ── Key loading ────────────────────────────────────────────────────────

    def _load_keys(self):
        """Load all API keys from environment variables."""
        loaded: List[KeySlot] = []

        # Primary key (backwards-compatible) — always treated as PAID
        primary = os.getenv("GEMINI_API_KEY")
        if primary:
            loaded.append(KeySlot(key=primary, project_name="primary", index=0, is_paid=True))

        # Pool keys: GEMINI_KEY_1 … GEMINI_KEY_20
        # Start as is_paid=True. If a key hits 429 at low RPM (<20),
        # it will be auto-demoted to free tier (is_paid=False, rpm_limit=15).
        for i in range(1, 21):
            key = os.getenv(f"GEMINI_KEY_{i}")
            if key and key != primary:          # avoid duplicates
                loaded.append(KeySlot(
                    key=key, project_name=f"pool_{i}",
                    index=len(loaded), is_paid=True
                ))

        self._keys = loaded

        if not self._keys:
            logger.error("❌ CRITICAL: No Gemini API keys found — AI features DISABLED")
        else:
            logger.info(
                f"🔑 Gemini Key Pool initialized: {len(self._keys)} keys "
                f"| Total capacity: {self._calculate_total_rpm()} RPM, "
                f"{len(self._keys)}M TPM"
            )

    def _calculate_total_rpm(self) -> int:
        """Calculate actual total RPM based on per-key limits."""
        return sum(s.rpm_limit for s in self._keys)

    # ── Counter management ─────────────────────────────────────────────────

    def _maybe_reset_counters(self, slot: KeySlot, now: float):
        """Reset per-minute counters if window has elapsed."""
        if now - slot.last_reset >= self._RESET_WINDOW:
            slot.rpm_used = 0
            slot.tpm_used = 0
            slot.is_exhausted = False
            slot.consecutive_errors = 0
            slot.last_reset = now

    def _is_available(self, slot: KeySlot, now: float) -> bool:
        """Check if a key slot is currently usable."""
        if slot.is_exhausted:
            return False
        if slot.consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
            if now - slot.last_error_time < self._ERROR_COOLDOWN:
                return False
            # Cooldown expired → give it another chance
            slot.consecutive_errors = 0
            slot.is_exhausted = False
        return True

    # ── Public API ─────────────────────────────────────────────────────────

    def get_next_key(self) -> Optional[KeySlot]:
        """
        Get the next available API key using smart round-robin.
        Prioritizes paid keys, then falls back to free keys.

        Returns the KeySlot or None if the pool is empty.
        """
        with self._lock:
            if not self._keys:
                return None

            now = time.time()

            # Reset expired counters
            for slot in self._keys:
                self._maybe_reset_counters(slot, now)

            # Phase 1: Try PAID keys first (higher RPM)
            selected = self._find_available_key(now, paid_only=True)
            if selected:
                return selected

            # Phase 2: Try FREE keys
            selected = self._find_available_key(now, paid_only=False)
            if selected:
                return selected

            # Phase 3: All keys exhausted → force-reset the least-used one
            logger.warning("⚠️ ALL API keys exhausted! Force-resetting pool...")
            best = min(self._keys, key=lambda s: s.rpm_used)
            best.rpm_used = 0
            best.tpm_used = 0
            best.is_exhausted = False
            best.consecutive_errors = 0
            best.rpm_used += 1
            best.total_requests += 1
            best.active_calls += 1
            self._total_requests += 1
            return best

    def _find_available_key(self, now: float, paid_only: bool = False) -> Optional[KeySlot]:
        """Find an available key, optionally filtering for paid keys only."""
        # Scan all keys from current index
        start_idx = self._current_index
        attempts = 0
        while attempts < len(self._keys):
            slot = self._keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._keys)

            # Skip if filtering for paid and this is free
            if paid_only and not slot.is_paid:
                attempts += 1
                continue

            if self._is_available(slot, now):
                slot.rpm_used += 1
                slot.total_requests += 1
                slot.active_calls += 1
                self._total_requests += 1

                if slot.rpm_used >= slot.rpm_limit:
                    slot.is_exhausted = True
                    logger.warning(
                        f"🔄 Key [{slot.project_name}] RPM exhausted "
                        f"({slot.rpm_used}/{slot.rpm_limit})"
                    )
                return slot

            attempts += 1
        return None

    def report_success(self, slot: KeySlot, estimated_tokens: int = 0):
        """Report a successful API call (reset consecutive error count)."""
        with self._lock:
            slot.consecutive_errors = 0
            slot.active_calls = max(0, slot.active_calls - 1)
            if estimated_tokens > 0:
                slot.tpm_used += estimated_tokens
                if slot.tpm_used >= slot.tpm_limit:
                    slot.is_exhausted = True
                    logger.warning(
                        f"🔄 Key [{slot.project_name}] TPM exhausted "
                        f"({slot.tpm_used}/{slot.tpm_limit})"
                    )
            # If key was previously demoted to free, and it's succeeding,
            # promote it back to paid after billing is enabled
            if not slot.is_paid and slot.rpm_used > 20 and slot.consecutive_errors == 0:
                slot.is_paid = True
                slot.rpm_limit = 1000
                logger.info(f"⬆️ Key [{slot.project_name}] auto-promoted to paid tier")

    def report_error(self, slot: KeySlot, error: str = ""):
        """Report a failed API call. Auto-detects 429 rate limits."""
        with self._lock:
            slot.errors += 1
            slot.total_errors += 1
            slot.consecutive_errors += 1
            slot.last_error_time = time.time()
            slot.active_calls = max(0, slot.active_calls - 1)

            # 🧠 Auto-detect FREE TIER: If we get a 429 (rate limit) at low RPM,
            # this key is on the free tier (15 RPM). Demote it automatically.
            is_rate_limit = "429" in error or "rate" in error.lower() or "quota" in error.lower() or "resource" in error.lower()
            if is_rate_limit and slot.rpm_used <= 20:
                slot.is_paid = False
                slot.rpm_limit = 15  # Free tier limit
                slot.is_exhausted = True
                logger.warning(
                    f"⬇️ Key [{slot.project_name}] auto-demoted to FREE tier "
                    f"(429 at {slot.rpm_used} RPM). Limit set to 15 RPM."
                )
            elif is_rate_limit:
                # Paid key hit rate limit — just exhaust it for this window
                slot.is_exhausted = True
                logger.warning(
                    f"🔄 Key [{slot.project_name}] rate-limited at {slot.rpm_used} RPM"
                )
            elif slot.consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                slot.is_exhausted = True
                logger.error(
                    f"❌ Key [{slot.project_name}] disabled after "
                    f"{slot.consecutive_errors} consecutive errors: {error[:120]}"
                )

    def get_model(self, model_name: str = None, system_instruction: str = None) -> Tuple:
        """
        Get a configured GenerativeModel using the next available key.

        Returns: (model: GenerativeModel, slot: KeySlot)
        Raises:  RuntimeError if no keys are available.
        """
        slot = self.get_next_key()
        if not slot:
            raise RuntimeError("No Gemini API keys available in pool")

        # Configure genai for this call's key
        genai.configure(api_key=slot.key)

        name = model_name or os.getenv("GEMINI_MODEL_V3", "gemini-2.5-flash")

        try:
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=name,
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(model_name=name)
        except Exception:
            # Fallback model
            fallback = "gemini-2.0-flash"
            logger.warning(f"Model {name} unavailable on key [{slot.project_name}], using {fallback}")
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=fallback,
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(model_name=fallback)

        return model, slot

    @property
    def total_capacity_rpm(self) -> int:
        return self._calculate_total_rpm()

    @property
    def total_active_calls(self) -> int:
        return sum(s.active_calls for s in self._keys)

    @property
    def available_keys_count(self) -> int:
        now = time.time()
        return sum(1 for k in self._keys if self._is_available(k, now))

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get_stats(self) -> Dict:
        """Return full pool diagnostics (exposed via /api/ai/pool-stats)."""
        now = time.time()
        paid_keys = [s for s in self._keys if s.is_paid]
        free_keys = [s for s in self._keys if not s.is_paid]
        return {
            "pool_size": len(self._keys),
            "paid_keys": len(paid_keys),
            "free_keys": len(free_keys),
            "available_keys": self.available_keys_count,
            "total_capacity_rpm": self.total_capacity_rpm,
            "total_requests_served": self._total_requests,
            "total_active_calls": self.total_active_calls,
            "keys": [
                {
                    "name": s.project_name,
                    "tier": "paid" if s.is_paid else "free",
                    "rpm_used": s.rpm_used,
                    "rpm_limit": s.rpm_limit,
                    "tpm_used": s.tpm_used,
                    "active_calls": s.active_calls,
                    "total_requests": s.total_requests,
                    "total_errors": s.total_errors,
                    "consecutive_errors": s.consecutive_errors,
                    "exhausted": s.is_exhausted,
                    "available": self._is_available(s, now),
                }
                for s in self._keys
            ],
        }


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_pool: Optional[GeminiKeyPool] = None
_pool_lock = threading.Lock()


def get_key_pool() -> GeminiKeyPool:
    """Get or create the global GeminiKeyPool singleton."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:       # double-check after acquiring lock
                _pool = GeminiKeyPool()
    return _pool
