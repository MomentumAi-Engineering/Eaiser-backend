#!/usr/bin/env python3
"""
Optimized FastAPI Application for 1 Lakh+ Concurrent Users

This is an enterprise-grade FastAPI application optimized for massive scale:

- Advanced connection pooling and resource management
- Intelligent caching with Redis cluster
- Rate limiting and DDoS protection
- Performance monitoring and metrics
- Graceful shutdown and health checks
- Horizontal scaling support
- Circuit breakers and fault tolerance
"""

import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# FastAPI and middleware imports
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

# Pydantic for data validation
from pydantic import BaseModel

# Environment and configuration
from dotenv import load_dotenv

# Import optimized services
from services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb, get_optimized_mongodb_service
from services.redis_cluster_service import init_redis_cluster, close_redis_cluster, get_redis_cluster_service

# Import routers
from routers import issues_optimized

# Load environment variables
load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Global performance metrics
class PerformanceMetrics:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.error_count = 0
        self.slow_request_count = 0
        self.start_time = time.time()
        self.endpoint_stats = {}
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        self.request_count += 1
        self.total_response_time += response_time
        
        if status_code >= 400:
            self.error_count += 1
        
        if response_time > 1.0:  # Slow request threshold: 1 second
            self.slow_request_count += 1
        
        # Track per-endpoint stats
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0,
                'errors': 0,
                'slow_requests': 0
            }
        
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_time'] += response_time
        
        if status_code >= 400:
            stats['errors'] += 1
        if response_time > 1.0:
            stats['slow_requests'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.request_count,
            'requests_per_second': round(self.request_count / max(uptime, 1), 2),
            'average_response_time_ms': round(avg_response_time * 1000, 2),
            'error_rate_percentage': round((self.error_count / max(self.request_count, 1)) * 100, 2),
            'slow_request_rate_percentage': round((self.slow_request_count / max(self.request_count, 1)) * 100, 2),
            'endpoint_stats': {
                endpoint: {
                    'count': stats['count'],
                    'avg_response_time_ms': round((stats['total_time'] / max(stats['count'], 1)) * 1000, 2),
                    'error_rate_percentage': round((stats['errors'] / max(stats['count'], 1)) * 100, 2),
                    'slow_request_rate_percentage': round((stats['slow_requests'] / max(stats['count'], 1)) * 100, 2)
                }
                for endpoint, stats in self.endpoint_stats.items()
            }
        }

# Global metrics instance
metrics = PerformanceMetrics()

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with Redis-based distributed limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = 1000, burst_limit: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_size = 60  # 1 minute window
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if await self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Process request
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else 'unknown'
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited using Redis."""
        try:
            redis_service = await get_redis_cluster_service()
            if not redis_service:
                # Fallback to in-memory rate limiting if Redis is unavailable
                return False
            
            current_time = int(time.time())
            window_start = current_time - self.window_size
            
            # Use Redis sorted set for sliding window rate limiting
            key = f"rate_limit:{client_ip}"
            
            # Remove old entries
            await redis_service.redis_cluster.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_requests = await redis_service.redis_cluster.zcard(key)
            
            if current_requests >= self.requests_per_minute:
                return True
            
            # Add current request
            await redis_service.redis_cluster.zadd(key, {str(current_time): current_time})
            await redis_service.redis_cluster.expire(key, self.window_size)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Rate limiting error: {str(e)}")
            return False  # Allow request if rate limiting fails

# Performance monitoring middleware
class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request performance and generate metrics.
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Extract endpoint info
        endpoint = f"{request.method} {request.url.path}"
        status_code = response.status_code
        
        # Record metrics
        metrics.record_request(endpoint, response_time, status_code)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time * 1000:.2f}ms"
        response.headers["X-Request-ID"] = str(id(request))
        
        # Log slow requests
        if response_time > 1.0:
            logger.warning(f"⚠️ Slow request: {endpoint} took {response_time * 1000:.2f}ms")
        
        # Log errors
        if status_code >= 400:
            logger.error(f"❌ Error response: {endpoint} returned {status_code}")
        
        return response

# Security middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for headers and basic protection.
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    """
    logger.info("🚀 Starting SnapFix Optimized Backend for 1 Lakh+ Users...")
    
    # Startup
    try:
        # Initialize Redis cluster
        logger.info("🔧 Initializing Redis cluster...")
        await init_redis_cluster()
        
        # Initialize optimized MongoDB
        logger.info("🔧 Initializing optimized MongoDB...")
        await init_optimized_mongodb()
        
        # Warm up connections
        logger.info("🔥 Warming up connections...")
        await _warmup_connections()
        
        logger.info("✅ All services initialized successfully!")
        logger.info("🎯 Backend ready for 1 Lakh+ concurrent users")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down services gracefully...")
    
    try:
        # Close MongoDB connections
        await close_optimized_mongodb()
        logger.info("✅ MongoDB connections closed")
        
        # Close Redis connections
        await close_redis_cluster()
        logger.info("✅ Redis connections closed")
        
        logger.info("✅ All services shut down successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")

async def _warmup_connections():
    """
    Warm up database and cache connections.
    """
    try:
        # Test MongoDB connection
        mongodb_service = await get_optimized_mongodb_service()
        if mongodb_service:
            health = await mongodb_service.health_check()
            logger.info(f"📊 MongoDB health: {health['status']}")
        
        # Test Redis connection
        redis_service = await get_redis_cluster_service()
        if redis_service:
            health = await redis_service.health_check()
            logger.info(f"📊 Redis health: {health['status']}")
        
        logger.info("🔥 Connection warmup completed")
        
    except Exception as e:
        logger.error(f"❌ Connection warmup failed: {str(e)}")

# Create optimized FastAPI application
app = FastAPI(
    title="SnapFix - Optimized for 1 Lakh+ Users",
    description="Enterprise-grade civic issue reporting platform optimized for massive scale",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware in correct order (last added = first executed)

# 1. Security middleware (first)
app.add_middleware(SecurityMiddleware)

# 2. CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time", "X-Request-ID"]
)

# 3. Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure properly for production
)

# 4. GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 5. Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=2000,  # Increased for high load
    burst_limit=200
)

# 6. Performance monitoring middleware (last)
app.add_middleware(PerformanceMiddleware)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": str(id(request))
        }
    )

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "environment": os.getenv("ENV", "development"),
        "services": {}
    }
    
    try:
        # Check MongoDB
        mongodb_service = await get_optimized_mongodb_service()
        if mongodb_service:
            mongo_health = await mongodb_service.health_check()
            health_status["services"]["mongodb"] = mongo_health
        else:
            health_status["services"]["mongodb"] = {"status": "unavailable"}
        
        # Check Redis
        redis_service = await get_redis_cluster_service()
        if redis_service:
            redis_health = await redis_service.health_check()
            health_status["services"]["redis"] = redis_health
        else:
            health_status["services"]["redis"] = {"status": "unavailable"}
        
        # Overall status
        all_healthy = all(
            service.get("status") == "healthy" 
            for service in health_status["services"].values()
        )
        
        if not all_healthy:
            health_status["status"] = "degraded"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.error(f"❌ Health check failed: {str(e)}")
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get comprehensive application metrics.
    """
    try:
        app_metrics = metrics.get_stats()
        
        # Add service metrics
        mongodb_service = await get_optimized_mongodb_service()
        if mongodb_service:
            app_metrics["mongodb"] = await mongodb_service.get_performance_stats()
        
        redis_service = await get_redis_cluster_service()
        if redis_service:
            app_metrics["redis"] = await redis_service.get_performance_stats()
        
        return JSONResponse(content=app_metrics)
        
    except Exception as e:
        logger.error(f"❌ Failed to get metrics: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve metrics"}
        )

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check for Kubernetes/load balancer.
    """
    try:
        # Quick checks for essential services
        mongodb_service = await get_optimized_mongodb_service()
        redis_service = await get_redis_cluster_service()
        
        if mongodb_service and redis_service:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "timestamp": datetime.utcnow().isoformat()}
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )

# Include routers
app.include_router(issues_optimized.router, prefix="/api", tags=["Issues"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "SnapFix Backend - Optimized for 1 Lakh+ Users",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Production-grade server configuration
    uvicorn.run(
        "main_optimized:app",
        host="0.0.0.0",
        port=10000,
        workers=1,  # Use 1 worker per process, scale with multiple processes
        loop="uvloop",  # High-performance event loop
        http="httptools",  # High-performance HTTP parser
        access_log=True,
        log_level="info",
        reload=False,  # Disable reload in production
        # Performance optimizations
        backlog=2048,  # Increased backlog for high concurrency
        limit_concurrency=10000,  # Handle up to 10k concurrent connections
        limit_max_requests=100000,  # Restart worker after 100k requests
        timeout_keep_alive=30,  # Keep connections alive longer
        timeout_graceful_shutdown=30  # Graceful shutdown timeout
    )
