from fastapi import FastAPI, HTTPException, Request
from app.utils.timing_middleware import TimingMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routes.issues import router as issues_router
from app.services.mongodb_service import init_db, close_db
from app.services.redis_service import init_redis, close_redis
import logging
import os
import uvicorn
import json
from pathlib import Path
from datetime import datetime

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

# Templates (for HTML pages)
templates = Jinja2Templates(directory="templates")

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
        # Load authorities from JSON file
        zip_code_authorities_path = Path("app/data/zip_code_authorities.json")
        
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

# NEW: JSON report endpoint for programmatic access
@app.get("/api/report")
async def get_report_json():
    try:
        # TODO: Replace with actual data fetching/processing
        report_data = []
        logger.info("📊 Report generated successfully")
        return {
            "status": "success",
            "message": "Report generated successfully",
            "data": report_data
        }
    except Exception as e:
        logger.error(f"💥 Error generating report JSON: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
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
