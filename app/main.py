import sys
import os
from pathlib import Path

# --- FORCE LOAD app/.env CORRECTLY ---
from dotenv import load_dotenv
# Load .env from app directory and override any global/system vars
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)

# Debug print (you can remove after testing)
print(f"[ENV CHECK] MONGO_URI = {os.getenv('MONGO_URI')}")
print(f"[ENV CHECK] ENVIRONMENT = {os.getenv('ENVIRONMENT')}")

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
    from routes.issues import router as issues_router
except ImportError:
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
    from services.mongodb_service import init_db, close_db
except ImportError:
    from app.services.mongodb_service import init_db, close_db

try:
    from services.redis_service import init_redis, close_redis
except ImportError:
    from app.services.redis_service import init_redis, close_redis

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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://easier-frontend.vercel.app",
        "https://www.eaiser.ai",
        "https://eaiser.ai",
        "https://eaiserai.io",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "X-Requested-With", "Origin",
        "Access-Control-Request-Method", "Access-Control-Request-Headers"
    ],
    expose_headers=["*"]
)

# Request timeout middleware to prevent hung requests
class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        try:
            default_timeout = int(os.environ.get("REQUEST_TIMEOUT_DEFAULT", "45"))
        except Exception:
            default_timeout = 45
        try:
            issues_timeout = int(os.environ.get("REQUEST_TIMEOUT_ISSUES", "60"))
        except Exception:
            issues_timeout = 60
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
        logger.info(f"📥 {request.method} {path} from {request.client.host}")
    try:
        response = await call_next(request)
        if should_log and response.status_code >= 400:
            logger.warning(f"⚠️ {response.status_code} for {request.method} {path}")
        return response
    except Exception as e:
        logger.error(f"💥 Error processing {request.method} {path}: {str(e)}", exc_info=True)
        raise

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

# Explicit wrappers to ensure endpoints exist even if router mounting varies
from fastapi import UploadFile, File, Query

@app.post("/api/ai/analyze-image")
async def analyze_image_endpoint(image: UploadFile = File(...), fast: bool = Query(False)):
    return await analyze_image_fn(image=image, fast=fast)

@app.post("/api/analyze-image")
async def analyze_image_alias_endpoint(image: UploadFile = File(...), fast: bool = Query(False)):
    return await analyze_image_alias_fn(image=image, fast=fast)

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
    await init_db()
    try:
        await init_redis()
        logger.info("✅ Eaiser AI backend started successfully with Redis caching")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down Eaiser AI backend...")
    await close_db()
    await close_redis()
    logger.info("✅ Shutdown completed successfully")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="debug", workers=8)
