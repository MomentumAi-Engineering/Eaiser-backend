import sys
import os
from pathlib import Path

# --- FORCE LOAD app/.env CORRECTLY ---
from dotenv import load_dotenv
# Load .env from app directory and override any global/system vars
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)

print("[ENV CHECK] Environment variables loaded.")


# Add current directory to Python path for Render deployment compatibility
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Also add parent directory to handle different deployment structures
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.services.websocket_manager import manager
import asyncio
import logging
import uvicorn
import json
from datetime import datetime
app = FastAPI(title="EAiSER AI Engine", version="2.0.0")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Logic to capture technical failures for debugging
    if response.status_code >= 400:
        # For 400/422, log headers to check Content-Type and field errors
        headers = dict(request.headers)
        logger.warning(f"❌ HTTP {response.status_code} on {request.method} {request.url.path}")
        logger.warning(f"   Headers: {headers}")
        if "content-type" in headers and "multipart" in headers["content-type"]:
            logger.info(f"   Note: This was a Multipart request (likely a file upload)")
            
    return response

# Ensure static directory exists before mounting
os.makedirs("static", exist_ok=True)
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup logging FIRST before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy pymongo background reconnect logs (DNS glitches are auto-retried)
logging.getLogger("pymongo.client").setLevel(logging.CRITICAL)
logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
logging.getLogger("pymongo.connection").setLevel(logging.WARNING)

# --- MIDDLEWARE & ROUTER IMPORTS ---
# We use 'app.x' as the primary import path for IDE/Linter compatibility.
# The project root in the IDE is 'EAiSER Ai - V2', and 'Eaiser-backend' is a subfolder.
# Note: If running locally from the 'app' directory, uvicorn might need PYTHONPATH='.'.

try:
    from app.utils.timing_middleware import TimingMiddleware
except ImportError:
    from utils.timing_middleware import TimingMiddleware

# Router imports
try:
    from app.routes.issues_optimized_v2 import router as issues_router
except ImportError:
    try:
        from routes.issues_optimized_v2 import router as issues_router
    except ImportError:
        try:
            from app.routes.issues import router as issues_router
        except ImportError:
            from routes.issues import router as issues_router

try:
    from app.api.reports import router as reports_router
except ImportError:
    try:
        from api.reports import router as reports_router
    except ImportError:
        reports_router = None


try:
    from app.api.ai import router as ai_router
    from app.api.ai import analyze_image as analyze_image_fn, analyze_image_alias as analyze_image_alias_fn
except ImportError:
    try:
        from api.ai import router as ai_router
        from api.ai import analyze_image as analyze_image_fn, analyze_image_alias as analyze_image_alias_fn
    except ImportError:
        ai_router = None
        analyze_image_fn = None
        analyze_image_alias_fn = None

try:
    from app.routes.admin_review import router as admin_review_router
except ImportError:
    try:
        from routes.admin_review import router as admin_review_router
    except ImportError:
        admin_review_router = None

try:
    from app.routes.admin_assignment import router as admin_assignment_router
except ImportError:
    try:
        from routes.admin_assignment import router as admin_assignment_router
    except ImportError:
        admin_assignment_router = None

try:
    from app.routes.admin_settings import router as admin_settings_router
except ImportError:
    from routes.admin_settings import router as admin_settings_router

try:
    from app.routes.analytics import router as analytics_router
except ImportError:
    try:
        from routes.analytics import router as analytics_router
    except ImportError:
        analytics_router = None

# Service imports
try:
    from services.mongodb_service import init_db, close_db
except ImportError:
    try:
        from app.services.mongodb_service import init_db, close_db
    except ImportError:
        init_db = close_db = None

try:
    from services.redis_service import init_redis, close_redis
except ImportError:
    try:
        from app.services.redis_service import init_redis, close_redis
    except ImportError:
        init_redis = close_redis = None

try:
    from services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb
except ImportError:
    try:
        from app.services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb
    except ImportError:
        init_optimized_mongodb = close_optimized_mongodb = None

try:
    from app.gov import gov_operations_router
except ImportError:
    try:
        from gov import gov_operations_router
    except ImportError:
        gov_operations_router = None

logger.info("🚀 Eaiser AI logging configuration initialized")



# Templates (for HTML pages)
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=template_dir)



# Request timeout middleware to prevent hung requests
class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        try:
            default_timeout = int(os.environ.get("REQUEST_TIMEOUT_DEFAULT", "45"))
        except Exception:
            default_timeout = 45
        try:
            issues_timeout = int(os.environ.get("REQUEST_TIMEOUT_ISSUES", "90"))
        except Exception:
            issues_timeout = 90
        try:
            ai_timeout = int(os.environ.get("REQUEST_TIMEOUT_AI", "60"))
        except Exception:
            ai_timeout = 60

        if path.startswith("/api/issues"):
            timeout = issues_timeout
        elif path.startswith("/api/ai/"):
            timeout = ai_timeout
        else:
            timeout = default_timeout

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timed out.",
                    "details": f"The server aborted the request after {timeout}s to stay responsive.",
                    "path": path
                }
            )

# Add timeout middleware before other middlewares
app.add_middleware(RequestTimeoutMiddleware)

# Production Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Anti-clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # HSTS (1 year, include subdomains)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Content Security Policy (refined for Google Auth/FedCM)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://accounts.google.com https://apis.google.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://accounts.google.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https: https://*.googleusercontent.com; "
            "media-src 'self' data: https: http:; "
            "connect-src 'self' https://eaiser.ai https://admin.eaiser.ai https://*.eaiser.ai "
            "https://accounts.google.com https://*.googleapis.com https://eaiser-backend-u8me.onrender.com http://localhost:8005 http://localhost:8000; "
            "frame-src 'self' https://accounts.google.com; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self' https://accounts.google.com"
        )
        # Required for Google Sign-In popups and One Tap
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Admin Panel Kill Switch Middleware
class AdminKillSwitchMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Check if this is an admin route
        if "/admin/" in path:
            # 1. Check ENV (Static override)
            admin_enabled_env = os.environ.get("ADMIN_PANEL_ENABLED", "true").lower()
            if admin_enabled_env == "false":
                return self._maintenance_response()
                
            # 2. Check DB (Dynamic toggle)
            try:
                # We use the service directly to avoid circular dependency issues
                from services.mongodb_service import get_db
                db = await get_db()
                settings = await db["settings"].find_one({"key": "maintenance_mode"})
                if settings and settings.get("value") is True:
                    # Exception: Allow login and settings routes so SA can turn it off
                    if not any(x in path for x in ["/login", "/2fa", "/maintenance-toggle", "/maintenance-status"]):
                        return self._maintenance_response()
            except Exception as e:
                # Fallback to allowing if DB check fails
                logger.error(f"Maintenance check failed: {e}")
                
        return await call_next(request)

    def _maintenance_response(self):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Admin panel is temporarily disabled for maintenance.",
                "status": "maintenance"
            }
        )

app.add_middleware(AdminKillSwitchMiddleware)

# Log all incoming requests
app.add_middleware(TimingMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    should_log = not any([
        path in ["/", "/health", "/favicon.ico", "/robots.txt"],
        path.startswith("/static/"),
        path.startswith("/assets/"),
        path.endswith((".ico", ".png", ".jpg", ".css", ".js"))
    ])
    
    if should_log:
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"📥 {request.method} {path} from {client_host}")
    
    try:
        response = await call_next(request)
        if should_log and response.status_code >= 400:
            if response.status_code == 401 and "/api/auth/login" in path:
                logger.warning(f"❌ AUTH FAILURE: 401 for {path}")
            logger.warning(f"⚠️ {response.status_code} for {request.method} {path}")
        return response
    except HTTPException:
        # Re-raise HTTPExceptions so they hit the official FastAPI handler
        raise
    except Exception as e:
        logger.error(f"💥 UNHANDLED ERROR in {request.method} {path}: {str(e)}", exc_info=True)
        # If we hit an error here, WE MUST still return a response or CORS will fail
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error during request processing"}
        )

# --------------------------------------------------------------------
# 🛡️ FINAL WRAPPERS (CORS MUST BE OUTSIDE EVERYTHING)
# --------------------------------------------------------------------

# Production CORS origins
CORS_ORIGINS = [
    "https://www.eaiser.ai",
    "https://eaiser.ai",
    "https://gov.eaiser.ai",
    "https://admin.eaiser.ai",
    "https://eaiserai.io",
    "https://easier-frontend.vercel.app",
    "http://easier-frontend.vercel.app",
    "https://eaiser-backend-u8me.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"http(s)?://(([a-zA-Z0-9-]+\.)?localhost|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.method} {request.url}: {str(exc)}", exc_info=True)
    # 🛡️ CRITICAL: Include CORS headers in error responses
    # Without this, browser blocks error responses and shows "Failed to fetch"
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in CORS_ORIGINS or origin.startswith("http://localhost"):
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
        }
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=headers
    )

@app.get("/api/proxy/places/autocomplete")
async def proxy_places_autocomplete(input: str, key: str):
    import urllib.request, json
    from urllib.parse import quote
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={quote(input)}&key={key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

@app.get("/api/proxy/places/details")
async def proxy_places_details(place_id: str, fields: str, key: str):
    import urllib.request, json
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields}&key={key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

@app.get("/api/proxy/geocode")
async def proxy_geocode(latlng: str = "", place_id: str = "", key: str = ""):
    import urllib.request, json
    from urllib.parse import quote
    api_key = key or os.environ.get("GOOGLE_API_KEY", "")
    if latlng:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={quote(latlng)}&key={api_key}"
    elif place_id:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?place_id={quote(place_id)}&key={api_key}"
    else:
        return {"results": [], "status": "INVALID_REQUEST"}
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

@app.api_route("/", methods=["GET", "HEAD"])
async def read_root():
    return {"message": "Eaiser AI backend is up and running!", "status": "healthy", "timestamp": datetime.now().isoformat()}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Dedicated health check endpoint for Render/load balancers."""
    return {"status": "healthy"}

# 🚀 Gemini Key Pool Monitoring Endpoint
@app.get("/api/ai/pool-stats")
async def gemini_pool_stats():
    """Live monitoring of the Gemini API key pool distribution."""
    try:
        from services.gemini_key_pool import get_key_pool
        pool = get_key_pool()
        stats = pool.get_stats()
        return {
            "status": "healthy",
            "pool": stats,
            "capacity_summary": {
                "total_keys": stats["pool_size"],
                "paid_keys": stats["paid_keys"],
                "free_keys": stats["free_keys"],
                "available_now": stats["available_keys"],
                "total_rpm_capacity": stats["total_capacity_rpm"],
                "total_active_calls": stats["total_active_calls"],
                "total_requests_served": stats["total_requests_served"],
                "estimated_reports_per_min": stats["total_capacity_rpm"] // 2,
                "note": "Free tier keys auto-detected via 429 errors. Enable billing to upgrade."
            }
        }
    except Exception as e:
        logger.error(f"Pool stats error: {e}")
        return {"status": "error", "error": str(e)}


@app.websocket("/ws/{user_email}")
async def websocket_endpoint(websocket: WebSocket, user_email: str):
    await manager.connect(websocket, user_email)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_email)
    except Exception as e:
        logger.error(f"WebSocket error for {user_email}: {e}")
        manager.disconnect(websocket, user_email)

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "service": "Eaiser AI Backend",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.environ.get("ENVIRONMENT", "production")
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/favicon.ico")
async def favicon():
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/db-health")
async def database_health_check():
    try:
        from services.mongodb_service import get_db
        db = await get_db()
        await db.command("ping")
        logger.info("✅ Database connection healthy")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"❌ Database health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")

@app.get("/api/authorities/{zip_code}")
async def get_authorities_by_zip_code(zip_code: str):
    try:
        zip_code_authorities_path = Path("data/zip_code_authorities.json")
        if not zip_code_authorities_path.exists():
            raise HTTPException(status_code=404, detail="Authorities data not found")
        with open(zip_code_authorities_path, "r") as f:
            authorities_data = json.load(f)
        if zip_code in authorities_data:
            authorities = authorities_data[zip_code]
        else:
            authorities = authorities_data.get("default", {})
        formatted_authorities = {}
        for auth_type, auth_list in authorities.items():
            formatted_authorities.setdefault(auth_type, []).extend(auth_list)
        return formatted_authorities
    except Exception as e:
        logger.error(f"Error fetching authorities for zip code {zip_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching authorities: {str(e)}")

# 🚀 OPTIMIZED REPORT GENERATION
@app.get("/api/report")
async def get_report_json(
    report_type: str = Query(default="performance"),
    format: str = Query(default="json"),
    cache_ttl: int = Query(default=300),
    use_cache: bool = Query(default=True)
):
    import time
    from services.report_generation_service import (
        HighPerformanceReportGenerator,
        ReportType,
        ReportConfig,
        create_report_generator
    )
    from core.database import get_db_dependency as get_database, get_redis_dependency as get_redis
    start_time = time.time()
    try:
        mongodb_client = await get_database()
        redis_client = await get_redis()
        generator = await create_report_generator(mongodb_client, redis_client)
        config = ReportConfig(
            report_type=ReportType(report_type.lower()),
            format=format.lower(),
            cache_ttl=cache_ttl if use_cache else 0,
            priority=1,
            filters={}
        )
        result = await generator.generate_report_fast(config)
        generation_time = time.time() - start_time
        logger.info(f"⚡ Report generated in {generation_time:.3f}s")
        return {
            "status": "success",
            "message": f"Report generated successfully in {generation_time:.3f} seconds",
            "data": result["data"],
            "metadata": {
                "report_type": report_type,
                "format": format,
                "generation_time": generation_time,
                "cache_hit": result.get("cache_hit", False),
                "generated_at": result["generated_at"].isoformat(),
                "performance_target_met": generation_time < 5.0
            }
        }
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"💥 Error generating report in {generation_time:.3f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.get("/report", response_class=HTMLResponse)
async def get_report_page(request: Request):
    try:
        report_data = []
        return templates.TemplateResponse(
            "report.html",
            {"request": request, "status": "success", "data": report_data}
        )
    except Exception as e:
        logger.error(f"Error generating report page: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report page: {str(e)}")

# Include API routes
if issues_router:
    app.include_router(issues_router, prefix="/api")
if reports_router:
    app.include_router(reports_router, prefix="/api/reports")
if ai_router:
    app.include_router(ai_router, prefix="/api")
if admin_review_router:
    app.include_router(admin_review_router, prefix="/api")  # Mounted at /api/admin/review
if admin_assignment_router:
    app.include_router(admin_assignment_router, prefix="/api") # Mounted at /api/admin/assignment
if admin_settings_router:
    app.include_router(admin_settings_router, prefix="/api") # Mounted at /api/admin/settings

# Include V5 Routing Engine Router
try:
    from app.routes.routing_v5 import router as routing_v5_router
except ImportError:
    try:
        from routes.routing_v5 import router as routing_v5_router
    except ImportError:
        routing_v5_router = None

if routing_v5_router:
    app.include_router(routing_v5_router, prefix="/api")
    logger.info("✅ V5 Routing Engine router mounted at /api/v5")

# Include Auth Router
try:
    import app.routes.auth as auth_mod
    auth_router = auth_mod.router
except ImportError:
    try:
        import routes.auth as auth_mod
        auth_router = auth_mod.router
    except ImportError:
        auth_router = None

if auth_router:
    app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

# Include Admin User Management
try:
    import app.routes.admin_users as admin_users_mod
    admin_users_router = admin_users_mod.router
except ImportError:
    try:
        import routes.admin_users as admin_users_mod
        admin_users_router = admin_users_mod.router
    except ImportError:
        admin_users_router = None

if admin_users_router:
    app.include_router(admin_users_router, prefix="/api") # Mounted at /api/admin/users

# Include Gov Auth Router
try:
    import app.gov.gov_auth as gov_auth_mod
    gov_auth_router = gov_auth_mod.router
except ImportError:
    try:
        import gov.gov_auth as gov_auth_mod
        gov_auth_router = gov_auth_mod.router
    except ImportError:
        gov_auth_router = None

if gov_auth_router:
    app.include_router(gov_auth_router, prefix="/api") # Mounted at /api/gov (auth)

# Include Gov Portal Content Router (Reports/Stats)
try:
    import app.gov.gov_portal as gov_portal_mod
    gov_portal_router = gov_portal_mod.router
except ImportError:
    try:
        import gov.gov_portal as gov_portal_mod
        gov_portal_router = gov_portal_mod.router
    except ImportError:
        gov_portal_router = None

if gov_portal_router:
    app.include_router(gov_portal_router, prefix="/api") # Mounted at /api/gov/portal

# Include Email Webhooks
try:
    import app.routes.email_webhook as webhook_mod
    email_webhook_router = webhook_mod.router
except ImportError:
    try:
        import routes.email_webhook as webhook_mod
        email_webhook_router = webhook_mod.router
    except ImportError:
        email_webhook_router = None

if email_webhook_router:
    app.include_router(email_webhook_router, prefix="/api/email", tags=["Email Webhooks"])

# Include Inquiry Router
try:
    import app.routes.inquiry as inquiry_mod
    inquiry_router = inquiry_mod.router
except ImportError:
    try:
        import routes.inquiry as inquiry_mod
        inquiry_router = inquiry_mod.router
    except ImportError:
        inquiry_router = None

if inquiry_router:
    app.include_router(inquiry_router, prefix="/api/inbound", tags=["Inquiries"])

if gov_operations_router:
    app.include_router(gov_operations_router, prefix="/api")

if analytics_router:
    app.include_router(analytics_router)

# Ensure endpoints exist via ai_router (mounted at /api)
# The redundant app.post wrappers here were causing 422 conflicts with the router definitions.


@app.get("/api/debug/routes")
async def debug_routes():
    from fastapi.routing import APIRoute
    return {
        "routes": [
            {
                "path": route.path,
                "name": route.name,
                "methods": list(route.methods or [])
            }
            for route in app.router.routes
            if isinstance(route, APIRoute)
        ]
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Eaiser AI backend server...")
    try:
        if init_db:
            await init_db()
    except Exception as e:
        logger.error(f"💥 Standard MongoDB Service initialization failed: {str(e)}")

    try:
        if init_redis:
            await init_redis()
            logger.info("✅ Redis caching initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {str(e)}")
    
    # Initialize Optimized MongoDB Service
    try:
        if init_optimized_mongodb:
            await init_optimized_mongodb()
            logger.info("✅ Optimized MongoDB Service initialized")
    except Exception as e:
        logger.error(f"❌ Optimized MongoDB Service initialization failed: {str(e)}")

    # Initialize Admin Login Monitoring indexes
    try:
        try:
            from services.admin_login_monitor import AdminLoginMonitor
        except ImportError:
            try:
                from app.services.admin_login_monitor import AdminLoginMonitor
            except ImportError:
                AdminLoginMonitor = None
                
        if AdminLoginMonitor:
            await AdminLoginMonitor.create_indexes()
            logger.info("✅ Admin login monitoring initialized")
        else:
            logger.warning("⚠️ Admin login monitor service not found")
    except Exception as e:
        logger.warning(f"⚠️ Admin login monitor index creation failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down Eaiser AI backend...")
    if close_db:
        await close_db()
    if close_redis:
        await close_redis()
    if close_optimized_mongodb:
        await close_optimized_mongodb()
    logger.info("✅ Shutdown completed successfully")

# Import authority service loader
try:
    from services.authority_service import load_mappings
except ImportError:
    try:
        from app.services.authority_service import load_mappings
    except ImportError:
        load_mappings = None

@app.on_event("startup")
async def load_authority_mappings():
    """Load authority mappings into memory."""
    try:
        if load_mappings:
            load_mappings()
    except Exception as e:
        logger.error(f"❌ Failed to load authority mappings: {e}")

# Attempt to use uvloop for high-performance event loop (Linux/Mac)
try:
    import uvloop
    uvloop.install()
    logger.info("Using uvloop for high-performance networking.")
except ImportError:
    logger.info("uvloop not available (Windows). Using standard asyncio loop.")

if __name__ == "__main__":
    import multiprocessing
    
    # Calculate optimal workers (CPU count + 1 or 2 usually decent for I/O bound)
    # For Windows, multiprocessing.cpu_count() is reliable
    try:
        cpu_cores = multiprocessing.cpu_count()
        workers = max(4, cpu_cores) # Ensure at least 4 workers
    except:
        workers = 4 # Fallback

    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting Eaiser AI High-Performance Node")
    logger.info(f"⚡ Configuration: {workers} Workers | Port: {port} | Loop: Auto")
    
    # RUN WITH MULTIPLE WORKERS (Load Balancing)
    # Use 'main:app' string to allow uvicorn to spawn worker processes
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        log_level="info", # Reduce from debug to info for production speed
        workers=workers,
        loop="auto",      # Will use uvloop if available
        http="auto",      # Will use httptools if available
        timeout_keep_alive=30,
        proxy_headers=True
    )
 
