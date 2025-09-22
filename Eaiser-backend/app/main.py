import sys
import os
from pathlib import Path

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
        # Fallback: create a simple timing middleware if import fails
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
    from services.mongodb_service import init_db, close_db
except ImportError:
    from app.services.mongodb_service import init_db, close_db

try:
    from services.redis_service import init_redis, close_redis
except ImportError:
    from app.services.redis_service import init_redis, close_redis

# Setup optimized logging with structured format
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce noise
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Reduce uvicorn noise
logging.getLogger("pymongo").setLevel(logging.WARNING)  # Reduce MongoDB noise
logging.getLogger("performance").setLevel(logging.INFO)  # Keep performance logs

# Create FastAPI app
app = FastAPI(title="Eaiser AI Backend")

# Templates (for HTML pages) - Using absolute path for proper template loading
import os
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=template_dir)

# Enable CORS for frontend access with explicit headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://easier-frontend.vercel.app",
        "https://www.eaiser.ai",
        "https://eaiser.ai",     # Frontend URL on Render
        "https://snapfixai.io",              # Frontend URL on SnapFix AI
        "http://localhost:5173",              # Local development (Vite default)
        "http://localhost:3000",              # Local development (React default)
        "http://localhost:3001"               # Local development (Alternative port)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"]
)

# Log all incoming requests
app.add_middleware(TimingMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Filter out noisy health check and static file requests
    path = request.url.path
    should_log = not any([
        path in ["/", "/health", "/favicon.ico", "/robots.txt"],
        path.startswith("/static/"),
        path.startswith("/assets/"),
        path.endswith(".ico"),
        path.endswith(".png"),
        path.endswith(".jpg"),
        path.endswith(".css"),
        path.endswith(".js")
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

# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception in {request.method} {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Root route to test backend
@app.get("/")
async def read_root():
    return {"message": "Eaiser AI backend is up and running!", "status": "healthy", "timestamp": datetime.now().isoformat()}

# Health check endpoint for Render
@app.get("/health")
async def health_check():
    try:
        # Basic health check
        return {
            "status": "healthy",
            "service": "Eaiser AI Backend",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.environ.get("ENVIRONMENT", "production")
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Favicon route
@app.get("/favicon.ico")
async def favicon():
    logger.debug("Favicon requested")
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    logger.warning("Favicon file not found")
    raise HTTPException(status_code=404, detail="Favicon not found")

# Database health check endpoint
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

# Authorities endpoint
@app.get("/api/authorities/{zip_code}")
async def get_authorities_by_zip_code(zip_code: str):
    try:
        # Load authorities from JSON file - fix path to be relative to current directory
        zip_code_authorities_path = Path("data/zip_code_authorities.json")
        
        if not zip_code_authorities_path.exists():
            raise HTTPException(status_code=404, detail="Authorities data not found")

        with open(zip_code_authorities_path, "r") as f:
            authorities_data = json.load(f)

        # Get authorities for the specified zip code or use default
        if zip_code in authorities_data:
            authorities = authorities_data[zip_code]
        else:
            authorities = authorities_data.get("default", {})

        # Format authorities as a list of objects by type
        formatted_authorities = {}
        for auth_type, auth_list in authorities.items():
            formatted_authorities.setdefault(auth_type, []).extend(auth_list)

        return formatted_authorities

    except Exception as e:
        logger.error(f"Error fetching authorities for zip code {zip_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching authorities: {str(e)}")

# 🚀 OPTIMIZED: High-Performance Report Generation Endpoint
@app.get("/api/report")
async def get_report_json(
    report_type: str = Query(default="performance", description="Type of report to generate"),
    format: str = Query(default="json", description="Report format (json/html/csv)"),
    cache_ttl: int = Query(default=300, description="Cache TTL in seconds"),
    use_cache: bool = Query(default=True, description="Use cached results if available")
):
    """
    🚀 Ultra-fast report generation endpoint
    Performance target: < 5 seconds
    """
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
        # Initialize high-performance report generator
        mongodb_client = await get_database()
        redis_client = await get_redis()
        generator = await create_report_generator(mongodb_client, redis_client)
        
        # Create optimized report configuration
        config = ReportConfig(
            report_type=ReportType(report_type.lower()),
            format=format.lower(),
            cache_ttl=cache_ttl if use_cache else 0,  # Disable cache if requested
            priority=1,  # High priority for fast generation
            filters={}
        )
        
        # Generate report with maximum speed optimization
        result = await generator.generate_report_fast(config)
        
        generation_time = time.time() - start_time
        
        logger.info(f"⚡ Report generated in {generation_time:.3f}s (Target: <5s)")
        
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)} (Time: {generation_time:.3f}s)"
        )

# UPDATED: HTML report page for browser access
@app.get("/report", response_class=HTMLResponse)
async def get_report_page(request: Request):
    logger.debug("Report HTML endpoint accessed")
    try:
        # TODO: Replace with actual data fetching/processing
        report_data = []
        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "status": "success",
                "message": "Report generated successfully",
                "data": report_data
            }
        )
    except Exception as e:
        logger.error(f"Error generating report page: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report page: {str(e)}"
        )

# Include API routes
app.include_router(issues_router, prefix="/api")
app.include_router(reports_router, prefix="/api/reports")

# Log startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Eaiser AI backend server...")
    await init_db()
    
    # Initialize Redis in background to avoid blocking startup
    try:
        await init_redis()  # Initialize Redis caching service
        logger.info("✅ Eaiser AI backend started successfully with Redis caching")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {str(e)}. Continuing without Redis.")
        logger.info("✅ Eaiser AI backend started successfully (Redis unavailable)")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 Shutting down Eaiser AI backend...")
    await close_db()
    await close_redis()  # Close Redis connection
    logger.info("✅ Shutdown completed successfully")

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render injects PORT
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        log_level="debug",
        workers=8   
    )
