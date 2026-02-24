"""
Admin Login Monitoring Service
- Dedicated admin_login_logs collection
- IP-based blocking: 3 failed attempts in 5 minutes → 15 minute block
- Real IP extraction with Cloudflare support
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Request

logger = logging.getLogger(__name__)


class AdminLoginMonitor:
    """
    Monitors admin login attempts and enforces IP-based blocking.
    Uses a dedicated 'admin_login_logs' MongoDB collection.
    """

    FAILED_ATTEMPT_THRESHOLD = 3  # Block after 3 failures
    FAILED_WINDOW_MINUTES = 5  # Within 5 minutes
    BLOCK_DURATION_MINUTES = 15  # Block for 15 minutes

    @staticmethod
    def extract_real_ip(request: Request) -> str:
        """
        Extract real client IP behind Cloudflare/proxy.
        Priority: CF-Connecting-IP > X-Real-IP > X-Forwarded-For > client.host
        """
        # Cloudflare sends real IP in this header
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()

        # Nginx/reverse proxy
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Standard proxy header (take first IP = original client)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Direct connection
        return request.client.host if request.client else "unknown"

    @staticmethod
    async def _get_collection():
        """Get the admin_login_logs collection."""
        try:
            from services.mongodb_optimized_service import get_optimized_mongodb_service
            mongo_service = await get_optimized_mongodb_service()
            if mongo_service:
                return await mongo_service.get_collection("admin_login_logs")
        except Exception as e:
            logger.error(f"Failed to get admin_login_logs collection: {e}")
        return None

    @staticmethod
    async def _get_blocked_ips_collection():
        """Get the blocked_ips collection."""
        try:
            from services.mongodb_optimized_service import get_optimized_mongodb_service
            mongo_service = await get_optimized_mongodb_service()
            if mongo_service:
                return await mongo_service.get_collection("blocked_ips")
        except Exception as e:
            logger.error(f"Failed to get blocked_ips collection: {e}")
        return None

    @staticmethod
    async def log_attempt(
        email: str,
        ip: str,
        user_agent: str,
        success: bool,
        failure_reason: Optional[str] = None
    ):
        """
        Log a login attempt to admin_login_logs collection.
        If 3+ failures in 5 min from same IP, block the IP for 15 min.
        """
        try:
            collection = await AdminLoginMonitor._get_collection()
            if not collection:
                return

            log_entry = {
                "email": email,
                "ip": ip,
                "user_agent": user_agent,
                "success": success,
                "timestamp": datetime.utcnow(),
            }
            if failure_reason:
                log_entry["failure_reason"] = failure_reason

            await collection.insert_one(log_entry)

            # Check if IP should be blocked (on failure only)
            if not success:
                window_start = datetime.utcnow() - timedelta(
                    minutes=AdminLoginMonitor.FAILED_WINDOW_MINUTES
                )
                recent_failures = await collection.count_documents({
                    "ip": ip,
                    "success": False,
                    "timestamp": {"$gte": window_start}
                })

                if recent_failures >= AdminLoginMonitor.FAILED_ATTEMPT_THRESHOLD:
                    await AdminLoginMonitor._block_ip(ip)
                    logger.warning(
                        f"🚫 IP {ip} BLOCKED after {recent_failures} failed login attempts"
                    )

        except Exception as e:
            logger.error(f"Error logging admin login attempt: {e}")

    @staticmethod
    async def _block_ip(ip: str):
        """Block an IP address for BLOCK_DURATION_MINUTES."""
        try:
            collection = await AdminLoginMonitor._get_blocked_ips_collection()
            if not collection:
                return

            blocked_until = datetime.utcnow() + timedelta(
                minutes=AdminLoginMonitor.BLOCK_DURATION_MINUTES
            )

            await collection.update_one(
                {"ip": ip},
                {
                    "$set": {
                        "ip": ip,
                        "blocked_until": blocked_until,
                        "reason": "excessive_failed_logins",
                        "blocked_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error blocking IP {ip}: {e}")

    @staticmethod
    async def is_ip_blocked(ip: str) -> Dict:
        """
        Check if an IP is currently blocked.
        Returns: {"blocked": bool, "remaining_minutes": int, "blocked_until": datetime}
        """
        try:
            collection = await AdminLoginMonitor._get_blocked_ips_collection()
            if not collection:
                return {"blocked": False}

            record = await collection.find_one({"ip": ip})
            if not record:
                return {"blocked": False}

            blocked_until = record.get("blocked_until")
            if blocked_until and blocked_until > datetime.utcnow():
                remaining = (blocked_until - datetime.utcnow()).total_seconds() / 60
                return {
                    "blocked": True,
                    "remaining_minutes": int(remaining) + 1,
                    "blocked_until": blocked_until
                }

            # Block expired — clean up
            await collection.delete_one({"ip": ip})
            return {"blocked": False}

        except Exception as e:
            logger.error(f"Error checking IP block status for {ip}: {e}")
            return {"blocked": False}  # Fail open

    @staticmethod
    async def create_indexes():
        """Create TTL and compound indexes for efficient querying."""
        try:
            logs_collection = await AdminLoginMonitor._get_collection()
            if logs_collection:
                # Compound index for IP + success + timestamp queries
                await logs_collection.create_index(
                    [("ip", 1), ("success", 1), ("timestamp", -1)],
                    name="ip_success_timestamp_idx"
                )
                # TTL index: auto-delete logs older than 30 days
                await logs_collection.create_index(
                    "timestamp",
                    expireAfterSeconds=30 * 24 * 60 * 60,
                    name="ttl_30d_idx"
                )
                logger.info("✅ admin_login_logs indexes created")

            blocked_collection = await AdminLoginMonitor._get_blocked_ips_collection()
            if blocked_collection:
                # TTL index: auto-delete expired blocks
                await blocked_collection.create_index(
                    "blocked_until",
                    expireAfterSeconds=0,
                    name="ttl_block_expiry_idx"
                )
                logger.info("✅ blocked_ips indexes created")

        except Exception as e:
            logger.error(f"Error creating admin login monitor indexes: {e}")
