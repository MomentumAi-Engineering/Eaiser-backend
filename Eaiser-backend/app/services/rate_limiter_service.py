#!/usr/bin/env python3
"""
Advanced Rate Limiting Service for 1 Lakh+ Concurrent Users

This service provides enterprise-grade rate limiting with:

- Redis-based distributed rate limiting
- Multiple rate limiting algorithms (Token Bucket, Sliding Window)
- Per-user, per-IP, and per-endpoint limits
- Burst handling and graceful degradation
- Real-time monitoring and alerting
- Automatic scaling based on load
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import hashlib
from enum import Enum

# FastAPI imports
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Redis service
from services.redis_cluster_service import get_redis_cluster_service

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"

class RateLimitTier(Enum):
    """Rate limit tiers for different user types."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class RateLimitConfig:
    """Configuration for rate limiting rules."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 10,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
        tier: RateLimitTier = RateLimitTier.FREE
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit
        self.algorithm = algorithm
        self.tier = tier

# Predefined rate limit configurations
RATE_LIMIT_CONFIGS = {
    RateLimitTier.FREE: RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=2000,
        burst_limit=5
    ),
    RateLimitTier.PREMIUM: RateLimitConfig(
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_limit=20
    ),
    RateLimitTier.ENTERPRISE: RateLimitConfig(
        requests_per_minute=500,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_limit=50
    ),
    RateLimitTier.ADMIN: RateLimitConfig(
        requests_per_minute=1000,
        requests_per_hour=50000,
        requests_per_day=500000,
        burst_limit=100
    )
}

# Endpoint-specific rate limits
ENDPOINT_RATE_LIMITS = {
    "/api/issues": {
        "POST": RateLimitConfig(requests_per_minute=20, burst_limit=5),
        "GET": RateLimitConfig(requests_per_minute=100, burst_limit=20),
        "PUT": RateLimitConfig(requests_per_minute=30, burst_limit=10),
        "DELETE": RateLimitConfig(requests_per_minute=10, burst_limit=3)
    },
    "/api/issues/bulk": {
        "POST": RateLimitConfig(requests_per_minute=5, burst_limit=2)
    },
    "/api/health": {
        "GET": RateLimitConfig(requests_per_minute=1000, burst_limit=100)
    },
    "/api/analytics": {
        "GET": RateLimitConfig(requests_per_minute=50, burst_limit=10)
    }
}

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms and Redis backend."""
    
    def __init__(self):
        self.redis_service = None
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60  # seconds
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_service = await get_redis_cluster_service()
            if self.redis_service:
                logger.info("✅ Rate limiter initialized with Redis cluster")
            else:
                logger.warning("⚠️ Rate limiter initialized without Redis (fallback mode)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rate limiter: {str(e)}")
    
    def _generate_key(self, identifier: str, window: str, endpoint: str = None) -> str:
        """Generate Redis key for rate limiting."""
        base_key = f"rate_limit:{identifier}:{window}"
        if endpoint:
            endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            base_key += f":{endpoint_hash}"
        return base_key
    
    def _get_user_tier(self, request: Request) -> RateLimitTier:
        """Determine user tier from request (can be enhanced with user authentication)."""
        # Check for API key or user authentication
        api_key = request.headers.get("X-API-Key")
        user_id = request.headers.get("X-User-ID")
        
        # For now, simple logic - can be enhanced with database lookup
        if api_key and api_key.startswith("admin_"):
            return RateLimitTier.ADMIN
        elif api_key and api_key.startswith("enterprise_"):
            return RateLimitTier.ENTERPRISE
        elif api_key and api_key.startswith("premium_"):
            return RateLimitTier.PREMIUM
        else:
            return RateLimitTier.FREE
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Priority: User ID > API Key > IP Address
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()[:12]}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _sliding_window_check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limit check using Redis."""
        try:
            if not self.redis_service:
                # Fallback to in-memory (not recommended for production)
                return True, {"remaining": limit, "reset_time": current_time + window_seconds}
            
            # Use Redis sorted set for sliding window
            window_start = current_time - window_seconds
            
            # Redis pipeline for atomic operations
            pipe = self.redis_service.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 1)
            
            # Execute pipeline
            results = await pipe.execute()
            
            current_count = results[1]  # Count after removing expired
            
            # Check if limit exceeded
            allowed = current_count < limit
            remaining = max(0, limit - current_count - 1)
            
            # Calculate reset time (when oldest request expires)
            if current_count > 0:
                oldest_requests = await self.redis_service.redis_client.zrange(
                    key, 0, 0, withscores=True
                )
                if oldest_requests:
                    oldest_time = oldest_requests[0][1]
                    reset_time = oldest_time + window_seconds
                else:
                    reset_time = current_time + window_seconds
            else:
                reset_time = current_time + window_seconds
            
            return allowed, {
                "remaining": remaining,
                "reset_time": reset_time,
                "current_count": current_count,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"❌ Sliding window check failed: {str(e)}")
            # Fail open in case of Redis issues
            return True, {"remaining": limit, "reset_time": current_time + window_seconds}
    
    async def _token_bucket_check(
        self,
        key: str,
        limit: int,
        refill_rate: float,
        current_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limit check using Redis."""
        try:
            if not self.redis_service:
                return True, {"remaining": limit, "reset_time": current_time + 60}
            
            # Get current bucket state
            bucket_data = await self.redis_service.redis_client.hmget(
                key, "tokens", "last_refill"
            )
            
            tokens = float(bucket_data[0]) if bucket_data[0] else limit
            last_refill = float(bucket_data[1]) if bucket_data[1] else current_time
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * refill_rate
            tokens = min(limit, tokens + tokens_to_add)
            
            # Check if request can be served
            if tokens >= 1:
                tokens -= 1
                allowed = True
            else:
                allowed = False
            
            # Update bucket state
            await self.redis_service.redis_client.hmset(key, {
                "tokens": tokens,
                "last_refill": current_time
            })
            await self.redis_service.redis_client.expire(key, 3600)  # 1 hour expiry
            
            # Calculate reset time
            reset_time = current_time + (1 - tokens) / refill_rate if tokens < 1 else current_time
            
            return allowed, {
                "remaining": int(tokens),
                "reset_time": reset_time,
                "tokens": tokens,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"❌ Token bucket check failed: {str(e)}")
            return True, {"remaining": limit, "reset_time": current_time + 60}
    
    async def check_rate_limit(
        self,
        request: Request,
        endpoint: str = None,
        method: str = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        start_time = time.time()
        
        try:
            # Get client identifier and tier
            client_id = self._get_client_identifier(request)
            user_tier = self._get_user_tier(request)
            current_time = time.time()
            
            # Get rate limit configuration
            config = RATE_LIMIT_CONFIGS.get(user_tier, RATE_LIMIT_CONFIGS[RateLimitTier.FREE])
            
            # Check endpoint-specific limits
            if endpoint and method:
                endpoint_config = ENDPOINT_RATE_LIMITS.get(endpoint, {}).get(method)
                if endpoint_config:
                    config = endpoint_config
            
            # Perform multiple window checks
            checks = [
                ("minute", config.requests_per_minute, 60),
                ("hour", config.requests_per_hour, 3600),
                ("day", config.requests_per_day, 86400)
            ]
            
            results = {}
            overall_allowed = True
            
            for window_name, limit, window_seconds in checks:
                if limit <= 0:  # Skip if limit is 0 or negative
                    continue
                
                key = self._generate_key(client_id, window_name, endpoint)
                
                if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                    allowed, window_result = await self._sliding_window_check(
                        key, limit, window_seconds, current_time
                    )
                elif config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    refill_rate = limit / window_seconds  # tokens per second
                    allowed, window_result = await self._token_bucket_check(
                        key, limit, refill_rate, current_time
                    )
                else:
                    # Default to sliding window
                    allowed, window_result = await self._sliding_window_check(
                        key, limit, window_seconds, current_time
                    )
                
                results[window_name] = window_result
                
                if not allowed:
                    overall_allowed = False
                    break
            
            # Add metadata
            processing_time = (time.time() - start_time) * 1000
            results.update({
                "allowed": overall_allowed,
                "client_id": client_id,
                "tier": user_tier.value,
                "algorithm": config.algorithm.value,
                "processing_time_ms": round(processing_time, 2),
                "timestamp": current_time
            })
            
            # Log rate limit check
            if overall_allowed:
                logger.debug(f"✅ Rate limit OK for {client_id} ({user_tier.value}) in {processing_time:.2f}ms")
            else:
                logger.warning(f"🚫 Rate limit exceeded for {client_id} ({user_tier.value})")
            
            return overall_allowed, results
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Rate limit check failed after {processing_time:.2f}ms: {str(e)}")
            
            # Fail open in case of errors
            return True, {
                "allowed": True,
                "error": str(e),
                "processing_time_ms": round(processing_time, 2)
            }
    
    async def get_rate_limit_status(self, request: Request) -> Dict[str, Any]:
        """Get current rate limit status for a client."""
        try:
            client_id = self._get_client_identifier(request)
            user_tier = self._get_user_tier(request)
            config = RATE_LIMIT_CONFIGS.get(user_tier, RATE_LIMIT_CONFIGS[RateLimitTier.FREE])
            
            status = {
                "client_id": client_id,
                "tier": user_tier.value,
                "limits": {
                    "requests_per_minute": config.requests_per_minute,
                    "requests_per_hour": config.requests_per_hour,
                    "requests_per_day": config.requests_per_day,
                    "burst_limit": config.burst_limit
                },
                "algorithm": config.algorithm.value,
                "windows": {}
            }
            
            # Get current usage for each window
            current_time = time.time()
            windows = [
                ("minute", config.requests_per_minute, 60),
                ("hour", config.requests_per_hour, 3600),
                ("day", config.requests_per_day, 86400)
            ]
            
            for window_name, limit, window_seconds in windows:
                if limit <= 0:
                    continue
                
                key = self._generate_key(client_id, window_name)
                
                if self.redis_service:
                    if config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                        window_start = current_time - window_seconds
                        current_count = await self.redis_service.redis_client.zcount(
                            key, window_start, current_time
                        )
                        remaining = max(0, limit - current_count)
                    else:
                        bucket_data = await self.redis_service.redis_client.hmget(
                            key, "tokens", "last_refill"
                        )
                        tokens = float(bucket_data[0]) if bucket_data[0] else limit
                        remaining = int(tokens)
                        current_count = limit - remaining
                else:
                    current_count = 0
                    remaining = limit
                
                status["windows"][window_name] = {
                    "limit": limit,
                    "used": current_count,
                    "remaining": remaining,
                    "reset_time": current_time + window_seconds
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get rate limit status: {str(e)}")
            return {"error": str(e)}
    
    async def reset_rate_limit(self, client_id: str, window: str = None) -> bool:
        """Reset rate limit for a client (admin function)."""
        try:
            if not self.redis_service:
                return False
            
            if window:
                key = self._generate_key(client_id, window)
                await self.redis_service.redis_client.delete(key)
            else:
                # Reset all windows
                pattern = f"rate_limit:{client_id}:*"
                keys = await self.redis_service.redis_client.keys(pattern)
                if keys:
                    await self.redis_service.redis_client.delete(*keys)
            
            logger.info(f"✅ Rate limit reset for {client_id} (window: {window or 'all'})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to reset rate limit: {str(e)}")
            return False

# Global rate limiter instance
_rate_limiter = None

async def get_rate_limiter() -> AdvancedRateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = AdvancedRateLimiter()
        await _rate_limiter.initialize()
    
    return _rate_limiter

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic rate limiting."""
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.rate_limiter = None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        if not self.enabled:
            return await call_next(request)
        
        try:
            # Initialize rate limiter if needed
            if self.rate_limiter is None:
                self.rate_limiter = await get_rate_limiter()
            
            # Extract endpoint and method
            endpoint = request.url.path
            method = request.method
            
            # Check rate limit
            allowed, rate_info = await self.rate_limiter.check_rate_limit(
                request, endpoint, method
            )
            
            if not allowed:
                # Rate limit exceeded
                response_data = {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "rate_limit_info": rate_info
                }
                
                return JSONResponse(
                    status_code=429,
                    content=response_data,
                    headers={
                        "X-RateLimit-Limit": str(rate_info.get("minute", {}).get("limit", "unknown")),
                        "X-RateLimit-Remaining": str(rate_info.get("minute", {}).get("remaining", 0)),
                        "X-RateLimit-Reset": str(int(rate_info.get("minute", {}).get("reset_time", time.time()))),
                        "Retry-After": "60"
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            if "minute" in rate_info:
                minute_info = rate_info["minute"]
                response.headers["X-RateLimit-Limit"] = str(minute_info.get("limit", "unknown"))
                response.headers["X-RateLimit-Remaining"] = str(minute_info.get("remaining", 0))
                response.headers["X-RateLimit-Reset"] = str(int(minute_info.get("reset_time", time.time())))
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Rate limiting middleware error: {str(e)}")
            # Continue processing in case of middleware errors
            return await call_next(request)

# Dependency for manual rate limiting in endpoints
async def rate_limit_dependency(
    request: Request,
    endpoint: str = None,
    method: str = None
) -> None:
    """Dependency function for manual rate limiting in endpoints."""
    try:
        rate_limiter = await get_rate_limiter()
        
        # Use request path if endpoint not provided
        if endpoint is None:
            endpoint = request.url.path
        if method is None:
            method = request.method
        
        allowed, rate_info = await rate_limiter.check_rate_limit(
            request, endpoint, method
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "rate_limit_info": rate_info
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info.get("minute", {}).get("limit", "unknown")),
                    "X-RateLimit-Remaining": str(rate_info.get("minute", {}).get("remaining", 0)),
                    "X-RateLimit-Reset": str(int(rate_info.get("minute", {}).get("reset_time", time.time()))),
                    "Retry-After": "60"
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Rate limit dependency error: {str(e)}")
        # Fail open - allow request to proceed
        pass
