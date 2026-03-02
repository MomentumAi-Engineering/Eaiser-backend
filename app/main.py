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

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import logging
import uvicorn
import json
from datetime import datetime

# Setup logging FIRST before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules with proper error handling
try:
    from utils.timing_middleware import TimingMiddleware
except ImportError:
    try:
        from app.utils.timing_middleware import TimingMiddleware
    except ImportError:
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware
        import time
        
        class TimingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                start_time = time.time()
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                return response

try:
    from routes.issues_optimized_v2 import router as issues_router
    logger.info("🚀 using optimized issues router v2")
except ImportError:
    try:
        from app.routes.issues_optimized_v2 import router as issues_router
        logger.info("🚀 using optimized issues router v2")
    except ImportError as e:
        logger.warning(f"⚠️ optimized router v2 not found, falling back to legacy: {e}")
        from app.routes.issues import router as issues_router

try:
    from api.reports import router as reports_router
except ImportError:
    from app.api.reports import router as reports_router

try:
    from api.ai import router as ai_router
    from api.ai import analyze_image as analyze_image_fn, analyze_image_alias as analyze_image_alias_fn
except ImportError:
    from app.api.ai import router as ai_router
    from app.api.ai import analyze_image as analyze_image_fn, analyze_image_alias as analyze_image_alias_fn

try:
    from routes.admin_review import router as admin_review_router
except ImportError:
    from app.routes.admin_review import router as admin_review_router

try:
    from routes.admin_assignment import router as admin_assignment_router
except ImportError:
    from app.routes.admin_assignment import router as admin_assignment_router

try:
    from routes.admin_settings import router as admin_settings_router
except ImportError:
    from app.routes.admin_settings import router as admin_settings_router

try:
    from services.mongodb_service import init_db, close_db
except ImportError:
    from app.services.mongodb_service import init_db, close_db

try:
    from services.redis_service import init_redis, close_redis
except ImportError:
    from app.services.redis_service import init_redis, close_redis

try:
    from services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb
except ImportError:
    from app.services.mongodb_optimized_service import init_optimized_mongodb, close_optimized_mongodb

# Setup optimized logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("performance").setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(title="Eaiser AI Backend")

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
            "connect-src 'self' https://eaiser.ai https://admin.eaiser.ai https://*.eaiser.ai "
            "https://accounts.google.com https://*.googleapis.com; "
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
            logger.warning(f"⚠️ {response.status_code} for {request.method} {path}")
        return response
    except Exception as e:
        logger.error(f"💥 Error processing {request.method} {path}: {str(e)}", exc_info=True)
        # If we hit an error here, WE MUST still return a response or CORS will fail
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error during request processing"}
        )

# --------------------------------------------------------------------
# 🛡️ FINAL WRAPPERS (CORS MUST BE OUTSIDE EVERYTHING)
# --------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://easier-frontend.vercel.app",
        "https://www.eaiser.ai",
        "https://eaiser.ai",
        "https://admin.eaiser.ai",
        "https://eaiserai.io",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://admin.localhost:5173",
        "http://admin.localhost:3000",
        "http://admin.localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.method} {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(exc)}"})

@app.get("/")
async def read_root():
    return {"message": "Eaiser AI backend is up and running!", "status": "healthy", "timestamp": datetime.now().isoformat()}

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
app.include_router(issues_router, prefix="/api")
app.include_router(reports_router, prefix="/api/reports")
app.include_router(ai_router, prefix="/api")
app.include_router(admin_review_router, prefix="/api")  # Mounted at /api/admin/review
app.include_router(admin_assignment_router, prefix="/api") # Mounted at /api/admin/assignment
app.include_router(admin_settings_router, prefix="/api") # Mounted at /api/admin/settings

# Import and include Auth Router
try:
    from routes.auth import router as auth_router
except ImportError:
    from app.routes.auth import router as auth_router

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

# Include Admin User Management
try:
    from routes.admin_users import router as admin_users_router
except ImportError:
    from app.routes.admin_users import router as admin_users_router

app.include_router(admin_users_router, prefix="/api") # Mounted at /api/admin/users

# Include Authority Action Router
try:
    from routes.authority_action import router as authority_action_router
except ImportError:
    from app.routes.authority_action import router as authority_action_router

app.include_router(authority_action_router, prefix="/api/authority-action", tags=["Authority Action"])

# Explicit wrappers to ensure endpoints exist even if router mounting varies
from fastapi import UploadFile, File

@app.post("/api/ai/analyze-image")
async def analyze_image_endpoint(image: UploadFile = File(...)):
    return await analyze_image_fn(image=image)

@app.post("/api/analyze-image")
async def analyze_image_alias_endpoint(image: UploadFile = File(...)):
    return await analyze_image_alias_fn(image=image)

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
        await init_db()
    except Exception as e:
        logger.error(f"💥 Standard MongoDB Service initialization failed: {str(e)}")

    try:
        await init_redis()
        logger.info("✅ Redis caching initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {str(e)}")
    
    # Initialize Optimized MongoDB Service
    try:
        await init_optimized_mongodb()
        logger.info("✅ Optimized MongoDB Service initialized")
    except Exception as e:
        logger.error(f"❌ Optimized MongoDB Service initialization failed: {str(e)}")

    # Initialize Admin Login Monitoring indexes
    try:
        from services.admin_login_monitor import AdminLoginMonitor
        await AdminLoginMonitor.create_indexes()
        logger.info("✅ Admin login monitoring initialized")
    except Exception as e:
        logger.warning(f"⚠️ Admin login monitor index creation failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down Eaiser AI backend...")
    await close_db()
    await close_redis()
    await close_optimized_mongodb()
    logger.info("✅ Shutdown completed successfully")

# Import authority service loader
try:
    from services.authority_service import load_mappings
except ImportError:
    from app.services.authority_service import load_mappings

@app.on_event("startup")
async def load_authority_mappings():
    """Load authority mappings into memory."""
    try:
        load_mappings()
    except Exception as e:
        logger.error(f"❌ Failed to load authority mappings: {e}")

# Attempt to use uvloop for high-performance event loop (Linux/Mac)
try:
    import uvloop
    uvloop.install()
    print("🚀 Using uvloop for high-performance networking.")
except ImportError:
    print("ℹ️ uvloop not installed or not supported (Windows). Using standard asyncio loop.")

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
 
