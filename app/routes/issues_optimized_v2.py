#!/usr/bin/env python3
"""
🚀 Ultra-Optimized Issues Router for 1 Lakh+ Concurrent Users

Performance Optimizations:
- Redis cluster caching for all operations
- Async background processing for heavy tasks
- Connection pooling and circuit breakers
- Batch operations and bulk processing
- Advanced rate limiting per endpoint
- Real-time performance monitoring
- Intelligent caching strategies
- Database query optimization

Author: Senior Full-Stack AI/ML Engineer
Target: 100,000+ concurrent users
"""

import asyncio
import logging
import time
import uuid
import base64
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# FastAPI imports
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

# Services - Using optimized versions
from services.mongodb_optimized_service import get_optimized_mongodb_service
from services.redis_cluster_service import get_redis_cluster_service
from services.ai_service import classify_issue
from services.ai_service_optimized import generate_report_optimized as generate_report
# import both send_email (existing) and send_formatted_ai_alert (new)
from services.email_service import send_email, send_formatted_ai_alert
from services.geocode_service import reverse_geocode, geocode_zip_code
from services.report_generation_service import build_unified_issue_json
from services.rate_limiter_service import AdvancedRateLimiter, RateLimitTier

# Utilities
from utils.location import get_authority, get_authority_by_zip_code
from utils.timezone import get_timezone_name
from utils.validators import validate_email, validate_zip_code
from utils.helpers import generate_report_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('issues_optimized.log')
    ]
)
logger = logging.getLogger(__name__)

# Create optimized router
router = APIRouter()

# Initialize rate limiter
rate_limiter = AdvancedRateLimiter()

# ========================================
# PERFORMANCE MONITORING
# ========================================
class PerformanceMetrics:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.db_queries = 0
        self.ai_processing_time = 0.0
        
    def record_request(self, response_time: float, error: bool = False):
        self.request_count += 1
        self.total_response_time += response_time
        if error:
            self.error_count += 1
            
    def record_cache_hit(self):
        self.cache_hits += 1
        
    def record_cache_miss(self):
        self.cache_misses += 1
        
    def record_db_query(self):
        self.db_queries += 1
        
    def record_ai_processing(self, processing_time: float):
        self.ai_processing_time += processing_time
        
    def get_stats(self) -> Dict[str, Any]:
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            "total_requests": self.request_count,
            "average_response_time_ms": round(avg_response_time * 1000, 2),
            "error_rate": round(error_rate * 100, 2),
            "cache_hit_rate": round(cache_hit_rate * 100, 2),
            "total_db_queries": self.db_queries,
            "total_ai_processing_time_s": round(self.ai_processing_time, 2)
        }

# Global performance metrics
performance_metrics = PerformanceMetrics()

# ========================================
# PYDANTIC MODELS
# ========================================
class IssueResponse(BaseModel):
    id: str
    message: str
    report: Optional[Dict] = None
    processing_time_ms: Optional[float] = None
    cached: Optional[bool] = False

class IssueStatusUpdate(BaseModel):
    status: str
    updated_by: Optional[str] = None
    notes: Optional[str] = None

class BulkIssueCreate(BaseModel):
    issues: List[Dict[str, Any]]
    batch_size: Optional[int] = 50

class IssueAnalytics(BaseModel):
    total_issues: int
    open_issues: int
    resolved_issues: int
    average_resolution_time: float
    issues_by_type: Dict[str, int]
    issues_by_severity: Dict[str, int]

class Issue(BaseModel):
    id: str = Field(..., alias="_id")
    address: str
    zip_code: Optional[str] = None
    latitude: float = 0.0
    longitude: float = 0.0
    issue_type: str
    severity: str
    status: str = "pending"
    report: Dict = {"message": "No report generated"}
    category: str = "public"
    priority: str = "Medium"
    timestamp: str
    user_email: Optional[str] = None
    authority_email: Optional[List[str]] = None
    authority_name: Optional[List[str]] = None
    processing_time_ms: Optional[float] = None
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

# ========================================
# CACHING UTILITIES
# ========================================
def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate consistent cache key from parameters"""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        if v is not None:
            key_parts.append(f"{k}:{str(v)}")
    return ":".join(key_parts)

async def get_cached_data(cache_key: str) -> Optional[Dict]:
    """Get data from Redis cache"""
    try:
        redis_service = await get_redis_cluster_service()
        if redis_service:
            # Use the new get_cache method with api_response cache type
            cached_data = await redis_service.get_cache('api_response', cache_key)
            if cached_data:
                performance_metrics.record_cache_hit()
                return cached_data
        performance_metrics.record_cache_miss()
        return None
    except Exception as e:
        logger.warning(f"Cache get failed for key {cache_key}: {e}")
        performance_metrics.record_cache_miss()
        return None

async def set_cached_data(cache_key: str, data: Dict, ttl: int = 300) -> bool:
    """Set data in Redis cache"""
    try:
        redis_service = await get_redis_cluster_service()
        if redis_service:
            # Use the new set_cache method with api_response cache type
            return await redis_service.set_cache('api_response', cache_key, data, ttl)
        return False
    except Exception as e:
        logger.warning(f"Cache set failed for key {cache_key}: {e}")
        return False

# ========================================
# RATE LIMITING DEPENDENCY
# ========================================
async def rate_limit_dependency(request: Request):
    """Rate limiting dependency for endpoints"""
    # Check rate limit using the correct method signature
    is_allowed, rate_limit_info = await rate_limiter.check_rate_limit(
        request=request,
        endpoint=str(request.url.path),
        method=request.method
    )
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )

# ========================================
# BACKGROUND PROCESSING
# ========================================
async def process_issue_background(
    issue_id: str,
    image_content: bytes,
    address: str,
    zip_code: Optional[str],
    latitude: float,
    longitude: float,
    user_email: Optional[str],
    description: str,
    category: str,
    severity: str,
    issue_type: str
):
    """Process issue in background for better performance"""
    start_time = time.time()
    try:
        # === Dispatch Guard import (local to function to avoid cold-start cost) ===
        # Hinglish: Yeh guard prank/false reports ko rokne me help karta hai.
        from services.dispatch_guard_service import AuthorityDispatchGuard

        # AI Classification (can be cached based on image hash)
        image_hash = hashlib.md5(image_content).hexdigest()
        cache_key = generate_cache_key(
            "ai_classification",
            image_hash=image_hash,
            description_hash=hashlib.md5((description or "").encode()).hexdigest()[:8]
        )
        
        cached_classification = await get_cached_data(cache_key)
        if cached_classification:
            issue_type = cached_classification["issue_type"]
            severity = cached_classification["severity"]
            confidence = cached_classification["confidence"]
            category = cached_classification["category"]
            priority = cached_classification["priority"]
            logger.info(f"Using cached AI classification for issue {issue_id}")
        else:
            ai_start = time.time()
            issue_type, severity, confidence, category, priority = await classify_issue(image_content, description or "")
            ai_time = time.time() - ai_start
            performance_metrics.record_ai_processing(ai_time)
            
            # Cache AI classification result
            classification_data = {
                "issue_type": issue_type,
                "severity": severity,
                "confidence": confidence,
                "category": category,
                "priority": priority
            }
            await set_cached_data(cache_key, classification_data, ttl=3600)  # Cache for 1 hour
        
        # Geocoding (can be cached based on zip_code or coordinates)
        final_address = address
        if zip_code:
            geocode_cache_key = generate_cache_key("geocode_zip", zip_code=zip_code)
            cached_geocode = await get_cached_data(geocode_cache_key)
            
            if cached_geocode:
                final_address = cached_geocode["address"]
                latitude = cached_geocode["latitude"]
                longitude = cached_geocode["longitude"]
            else:
                try:
                    geocode_result = await geocode_zip_code(zip_code)
                    final_address = geocode_result.get("address", address or "Unknown Address")
                    latitude = geocode_result.get("latitude", latitude)
                    longitude = geocode_result.get("longitude", longitude)
                    
                    # Cache geocoding result
                    await set_cached_data(geocode_cache_key, {
                        "address": final_address,
                        "latitude": latitude,
                        "longitude": longitude
                    }, ttl=86400)  # Cache for 24 hours
                except Exception as e:
                    logger.warning(f"Geocoding failed for zip {zip_code}: {e}")
        
        # Generate report (can be cached based on issue parameters)
        report_cache_key = generate_cache_key(
            "report",
            issue_type=issue_type,
            severity=severity,
            address=final_address,
            zip_code=zip_code,
            description_hash=hashlib.md5((description or "").encode()).hexdigest()[:8]
        )
        
        cached_report = await get_cached_data(report_cache_key)
        if cached_report:
            report = cached_report
            logger.info(f"Using cached report for issue {issue_id}")
        else:
            report = await generate_report(
                image_content=image_content,
                description=description or "",
                issue_type=issue_type,
                severity=severity,
                address=final_address,
                zip_code=zip_code,
                latitude=latitude,
                longitude=longitude,
                issue_id=issue_id,
                confidence=confidence,
                category=category,
                priority=priority
            )
            
            # Cache report (without sensitive data)
            try:
                report_to_cache = {k: v for k, v in report.items() if k not in ["template_fields"]}
                await set_cached_data(report_cache_key, report_to_cache, ttl=1800)  # Cache for 30 minutes
            except Exception as e:
                logger.warning(f"Failed to cache report for {issue_id}: {e}")
        
        # Get authorities (can be cached based on location)
        authority_cache_key = generate_cache_key(
            "authorities",
            zip_code=zip_code,
            issue_type=issue_type,
            category=category
        )
        
        cached_authorities = await get_cached_data(authority_cache_key)
        if cached_authorities:
            authority_data = cached_authorities
        else:
            try:
                authority_data = get_authority_by_zip_code(zip_code, issue_type, category) if zip_code else get_authority(final_address, issue_type, latitude, longitude, category)
                await set_cached_data(authority_cache_key, authority_data, ttl=3600)  # Cache for 1 hour
            except Exception as e:
                logger.warning(f"Failed to fetch authorities: {e}")
                authority_data = {
                    "responsible_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}],
                    "available_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
                }

        # === UPDATE LOCAL VARIABLES FROM FINAL REPORT ===
        # This ensures top-level DB fields match the AI's final analysis (e.g. if "Unknown" became "Pothole")
        try:
            overview = report.get("issue_overview", {})
            template_fields = report.get("template_fields", {})
            
            # Update issue_type if present and valid
            final_type = overview.get("type") or overview.get("issue_type")
            if final_type and final_type.lower() != "none":
                issue_type = final_type.lower().replace(" ", "_")
            
            # Update severity
            final_severity = overview.get("severity")
            if final_severity:
                severity = final_severity
                
            # Update confidence
            final_conf = overview.get("confidence")
            if final_conf is not None:
                confidence = float(final_conf)
                
            # Update category
            final_category = overview.get("category")
            if final_category:
                category = final_category.lower()
                
            # Update priority
            final_priority = template_fields.get("priority") or overview.get("priority")
            if final_priority:
                priority = final_priority
                
            logger.info(f"Finalized issue details for {issue_id}: Type={issue_type}, Severity={severity}, Conf={confidence}%")
        except Exception as e:
            logger.warning(f"Failed to update local variables from report for {issue_id}: {e}")

        # === Dispatch Guard Evaluation (before sending emails) ===
        try:
            # Basic metadata completeness: address or valid coords + image
            metadata_complete = bool(final_address) or (float(latitude) != 0.0 and float(longitude) != 0.0)
            metadata_complete = metadata_complete and bool(image_content)

            # Lightweight duplicate check: if classification cache hit on same image hash
            is_duplicate = bool(cached_classification)

            # Policy conflict placeholder: e.g., controlled bonfire in permitted area
            # Automatically detect if this is a controlled fire based on AI report content
            policy_conflict = False
            try:
                # Check summary and analysis for controlled fire keywords
                overview = report.get("issue_overview", {})
                ai_eval = report.get("ai_evaluation", {})
                combined_text = (str(overview.get("summary_explanation", "")) + " " + str(ai_eval.get("image_analysis", ""))).lower()
                
                controlled_fire_keywords = [
                    "controlled fire", "bonfire", "campfire", "bbq", "barbecue", "fire pit", 
                    "religious ceremony", "festival", "diwali", "candle", "stove", "hearth",
                    "chiminea", "grill", "cooking", "recreational fire", "fire ring", "holika", 
                    "dahan", "lohri", "ceremony", "ritual", "crowd", "people watching", 
                    "people standing", "spectators", "religious", "tradition", "custom"
                ]
                
                if any(keyword in combined_text for keyword in controlled_fire_keywords):
                    # Only flag as policy conflict if NO clear danger words are present
                    danger_words = ["out of control", "emergency", "wildfire", "house fire", "building fire", "structure fire", "explosion"]
                    if not any(danger in combined_text for danger in danger_words):
                        policy_conflict = True
                        logger.info(f"Policy conflict detected for issue {issue_id}: Controlled fire keywords found.")
            except Exception as e:
                logger.warning(f"Failed to check policy conflict for issue {issue_id}: {e}")

            # Reporter trust default (enhance later from user profile)
            reporter_trust_score = 50.0

            # Rate-limit breach (handled at endpoint via dependency, assume False here)
            rate_limit_breached = False

            guard_payload = {
                "issue_type": issue_type,
                "severity": severity,
                "priority": priority,
                "public_safety_risk": (report or {}).get("detailed_analysis", {}).get("public_safety_risk", ""),
                "image_analysis": (report or {}).get("ai_evaluation", {}).get("image_analysis", {}),
                "ai_confidence_percent": float(confidence or 0),
                "metadata_complete": metadata_complete,
                "is_duplicate": is_duplicate,
                "policy_conflict": policy_conflict,
                "reporter_trust_score": reporter_trust_score,
                "rate_limit_breached": rate_limit_breached,
            }

            guard = AuthorityDispatchGuard()
            decision = guard.evaluate(guard_payload)
            
            # Determine initial status based on guard decision
            final_status = "pending"
            if decision.action == "route_to_review_team":
                final_status = "needs_review"
            elif decision.action == "reject":
                final_status = "rejected"
            
            # Attach decision to report for downstream UI/ops
            try:
                if report is not None:
                    report["dispatch_decision"] = {
                        "action": decision.action,
                        "risk_score": decision.risk_score,
                        "fraud_score": decision.fraud_score,
                        "reasons": decision.reasons,
                        "suggested_next_steps": decision.suggested_next_steps,
                    }
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Dispatch guard evaluation failed for issue {issue_id}: {e}")
            decision = None
            final_status = "pending"

        # === NEW: Dispatch EAiSER formatted alert emails in background ===
        try:
            # Use the AI-generated formatted_report if present
            # Non-blocking: create background task so DB store is not delayed
            
            # Only send email if action is specifically 'auto_dispatch'
            if report and (decision and decision.action == "auto_dispatch"):
                try:
                    # fire-and-forget background send (safe)
                    asyncio.create_task(send_formatted_ai_alert(report, background=True))
                    logger.info(f"Dispatched EAiSER alert email task for issue {issue_id}")
                except Exception as e:
                    logger.warning(f"Failed to create email dispatch task for issue {issue_id}: {e}")
            else:
                # Hold, Reject, or Route to Review Team: do not email authority automatically
                if decision:
                    logger.info(f"Email dispatch skipped for issue {issue_id} due to guard action: {decision.action}")
        except Exception as e:
            logger.warning(f"Unexpected error while preparing to send EAiSER email for {issue_id}: {e}")
        
        # Store in optimized MongoDB
        mongodb_service = await get_optimized_mongodb_service()
        if mongodb_service:
            performance_metrics.record_db_query()
            
            # Prepare issue document
            # Compute time context for unified JSON
            try:
                timezone_name = get_timezone_name(latitude, longitude) or "UTC"
            except Exception:
                timezone_name = "UTC"
            timestamp_formatted = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

            # Build unified report JSON for consistent UI/Email rendering
            try:
                conf_val = 0.0
                try:
                    conf_val = float((report or {}).get("template_fields", {}).get("confidence", 0) or 0)
                except Exception:
                    conf_val = 0.0
                unified_report = build_unified_issue_json(
                    report=report or {},
                    issue_id=issue_id,
                    issue_type=issue_type,
                    category=category,
                    severity=severity,
                    priority=priority,
                    confidence=conf_val,
                    address=final_address,
                    zip_code=zip_code or "N/A",
                    latitude=latitude,
                    longitude=longitude,
                    timestamp_formatted=timestamp_formatted,
                    timezone_name=timezone_name,
                    department_type=None,
                    is_user_review=False,
                )
            except Exception:
                unified_report = {}
            issue_doc = {
                "_id": issue_id,
                "address": final_address,
                "zip_code": zip_code or "N/A",
                "latitude": latitude,
                "longitude": longitude,
                "issue_type": issue_type,
                "severity": severity,
                "category": category,
                "priority": priority,
                "status": final_status,
                "report": report,
                "unified_report": unified_report,
                "user_email": user_email,
                "authority_data": authority_data,
                "confidence": confidence,
                "timestamp": datetime.utcnow(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "image_hash": image_hash
            }
            
            # Store issue with optimized batch operation
            await mongodb_service.store_issue_optimized(issue_doc, image_content)
            
            logger.info(f"Issue {issue_id} processed successfully in background")
        
    except Exception as e:
        logger.error(f"Background processing failed for issue {issue_id}: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
    finally:
        # record request processing time (even on failure)
        try:
            performance_metrics.record_request(time.time() - start_time)
        except Exception:
            pass

# ========================================
# OPTIMIZED ENDPOINTS
# ========================================

@router.post("/issues", response_model=IssueResponse)
async def create_issue_optimized(
    background_tasks: BackgroundTasks,
    request: Request,
    image: UploadFile = File(...),
    address: str = Form(''),
    zip_code: Optional[str] = Form(None),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    user_email: Optional[str] = Form(None),
    description: str = Form(''),
    category: str = Form('public'),
    severity: str = Form('medium'),
    issue_type: str = Form('other'),
    _: None = Depends(rate_limit_dependency)
):
    """
    🚀 Ultra-optimized issue creation endpoint
    
    Features:
    - Async background processing
    - Intelligent caching
    - Fast response times
    - Automatic scaling
    """
    start_time = time.time()
    
    try:
        # Validate image format
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image content
        image_content = await image.read()
        if len(image_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Generate unique issue ID
        issue_id = str(uuid.uuid4())
        
        # Start background processing immediately
        background_tasks.add_task(
            process_issue_background,
            issue_id=issue_id,
            image_content=image_content,
            address=address,
            zip_code=zip_code,
            latitude=latitude,
            longitude=longitude,
            user_email=user_email,
            description=description,
            category=category,
            severity=severity,
            issue_type=issue_type
        )
        
        # Return immediate response
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        return IssueResponse(
            id=issue_id,
            message="Issue submitted successfully. Processing in background.",
            report={
                "issue_id": issue_id,
                "status": "processing",
                "estimated_completion": "2-5 seconds"
            },
            processing_time_ms=processing_time,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create issue: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to create issue: {str(e)}")

@router.get("/issues/{issue_id}", response_model=Issue)
async def get_issue_optimized(
    issue_id: str,
    request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """
    🔍 Get issue details with caching
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = generate_cache_key("issue", issue_id=issue_id)
        cached_issue = await get_cached_data(cache_key)
        
        if cached_issue:
            processing_time = (time.time() - start_time) * 1000
            performance_metrics.record_request(time.time() - start_time)
            
            # Convert datetime objects to strings for JSON serialization
            if 'timestamp' in cached_issue and hasattr(cached_issue['timestamp'], 'strftime'):
                cached_issue['timestamp'] = cached_issue['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Merge processing_time_ms into cached_issue to avoid duplicate parameter error
            cached_issue['processing_time_ms'] = processing_time
            return Issue(**cached_issue)
        
        # Get from database
        mongodb_service = await get_optimized_mongodb_service()
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        performance_metrics.record_db_query()
        issue_data = await mongodb_service.get_issue_by_id(issue_id)
        
        if not issue_data:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Cache the result
        await set_cached_data(cache_key, issue_data, ttl=300)  # Cache for 5 minutes
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        # Convert datetime objects to strings for JSON serialization
        if 'timestamp' in issue_data and hasattr(issue_data['timestamp'], 'strftime'):
            issue_data['timestamp'] = issue_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Merge processing_time_ms into issue_data to avoid duplicate parameter error
        issue_data['processing_time_ms'] = processing_time
        return Issue(**issue_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get issue {issue_id}: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to get issue: {str(e)}")

@router.get("/issues", response_model=List[Issue])
async def get_issues_optimized(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of issues to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of issues to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    issue_type: Optional[str] = Query(None, description="Filter by issue type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    _: None = Depends(rate_limit_dependency)
):
    """
    📋 Get issues list with advanced caching and filtering
    """
    start_time = time.time()
    
    try:
        # Generate cache key based on query parameters
        cache_key = generate_cache_key(
            "issues_list",
            skip=skip,
            limit=limit,
            status=status,
            issue_type=issue_type,
            severity=severity
        )
        
        # Check cache first
        cached_issues = await get_cached_data(cache_key)
        if cached_issues is not None:
            processing_time = (time.time() - start_time) * 1000
            performance_metrics.record_request(time.time() - start_time)
            
            # Convert datetime objects to strings for JSON serialization
            for issue_data in cached_issues:
                if 'timestamp' in issue_data and hasattr(issue_data['timestamp'], 'strftime'):
                    issue_data['timestamp'] = issue_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            return [
                Issue(**{**issue_data, 'processing_time_ms': processing_time})
                for issue_data in cached_issues
            ]
        
        # Get from database with optimized query
        mongodb_service = await get_optimized_mongodb_service()
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        performance_metrics.record_db_query()
        
        # Build filter query
        filter_query = {}
        if status:
            filter_query["status"] = status
        if issue_type:
            filter_query["issue_type"] = issue_type
        if severity:
            filter_query["severity"] = severity
        
        issues_data = await mongodb_service.get_issues_optimized(
            filter_query=filter_query,
            skip=skip,
            limit=limit
        )
        
        # Cache the result
        await set_cached_data(cache_key, issues_data, ttl=180)  # Cache for 3 minutes
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        # Convert datetime objects to strings for JSON serialization
        for issue_data in issues_data:
            if 'timestamp' in issue_data and hasattr(issue_data['timestamp'], 'strftime'):
                issue_data['timestamp'] = issue_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return [
            Issue(**{**issue_data, 'processing_time_ms': processing_time})
            for issue_data in issues_data
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get issues: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to get issues: {str(e)}")

@router.put("/issues/{issue_id}/status", response_model=IssueResponse)
async def update_issue_status_optimized(
    issue_id: str,
    status_update: IssueStatusUpdate,
    request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """
    ✏️ Update issue status with cache invalidation
    """
    start_time = time.time()
    
    try:
        mongodb_service = await get_optimized_mongodb_service()
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        performance_metrics.record_db_query()
        
        # Update in database
        update_data = {
            "status": status_update.status,
            "updated_at": datetime.utcnow()
        }
        
        if status_update.updated_by:
            update_data["updated_by"] = status_update.updated_by
        if status_update.notes:
            update_data["notes"] = status_update.notes
        
        success = await mongodb_service.update_issue_status(
            issue_id=issue_id,
            update_data=update_data
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Invalidate related caches
        cache_keys_to_invalidate = [
            generate_cache_key("issue", issue_id=issue_id),
            "issues_list:*"  # Invalidate all issues list caches
        ]
        
        redis_service = await get_redis_cluster_service()
        if redis_service:
            for cache_key in cache_keys_to_invalidate:
                if "*" in cache_key:
                    # Delete pattern-based keys
                    await redis_service.delete_cache_pattern('api_response', cache_key)
                else:
                    await redis_service.delete_cache('api_response', cache_key)
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        return IssueResponse(
            id=issue_id,
            message=f"Issue status updated to {status_update.status}",
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id}: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to update issue: {str(e)}")

@router.post("/issues/bulk", response_model=Dict[str, Any])
async def create_bulk_issues(
    bulk_request: BulkIssueCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """
    📦 Bulk issue creation for high-throughput scenarios
    """
    start_time = time.time()
    
    try:
        if len(bulk_request.issues) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 issues per batch")
        
        issue_ids = []
        batch_size = min(bulk_request.batch_size, 50)
        
        # Process in batches
        for i in range(0, len(bulk_request.issues), batch_size):
            batch = bulk_request.issues[i:i + batch_size]
            batch_ids = []
            
            for issue_data in batch:
                issue_id = str(uuid.uuid4())
                batch_ids.append(issue_id)
                
                # Add to background processing
                background_tasks.add_task(
                    process_issue_background,
                    issue_id=issue_id,
                    image_content=base64.b64decode(issue_data.get("image_base64", "")),
                    address=issue_data.get("address", ""),
                    zip_code=issue_data.get("zip_code"),
                    latitude=issue_data.get("latitude", 0.0),
                    longitude=issue_data.get("longitude", 0.0),
                    user_email=issue_data.get("user_email"),
                    description=issue_data.get("description", ""),
                    category=issue_data.get("category", "public"),
                    severity=issue_data.get("severity", "medium"),
                    issue_type=issue_data.get("issue_type", "other")
                )
            
            issue_ids.extend(batch_ids)
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        return {
            "message": f"Successfully queued {len(issue_ids)} issues for processing",
            "issue_ids": issue_ids,
            "processing_time_ms": processing_time,
            "estimated_completion": "5-15 seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create bulk issues: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to create bulk issues: {str(e)}")

@router.get("/issues/analytics/summary", response_model=IssueAnalytics)
async def get_issues_analytics(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days for analytics"),
    _: None = Depends(rate_limit_dependency)
):
    """
    📊 Get issues analytics with caching
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = generate_cache_key("analytics", days=days, date=datetime.utcnow().date())
        cached_analytics = await get_cached_data(cache_key)
        
        if cached_analytics:
            processing_time = (time.time() - start_time) * 1000
            performance_metrics.record_request(time.time() - start_time)
            return IssueAnalytics(**cached_analytics)
        
        # Get from database
        mongodb_service = await get_optimized_mongodb_service()
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        performance_metrics.record_db_query()
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        analytics_data = await mongodb_service.get_analytics_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        # Cache the result
        await set_cached_data(cache_key, analytics_data, ttl=1800)  # Cache for 30 minutes
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        return IssueAnalytics(**analytics_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.delete("/issues/{issue_id}")
async def delete_issue_optimized(
    issue_id: str,
    request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """
    🗑️ Delete issue with cache cleanup
    """
    start_time = time.time()
    
    try:
        mongodb_service = await get_optimized_mongodb_service()
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        performance_metrics.record_db_query()
        
        # Soft delete (mark as deleted)
        success = await mongodb_service.soft_delete_issue(issue_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Clean up caches
        redis_service = await get_redis_cluster_service()
        if redis_service:
            cache_keys_to_delete = [
                generate_cache_key("issue", issue_id=issue_id),
                "issues_list:*",
                "analytics:*"
            ]
            
            for cache_key in cache_keys_to_delete:
                if "*" in cache_key:
                    await redis_service.delete_cache_pattern('api_response', cache_key)
                else:
                    await redis_service.delete_cache('api_response', cache_key)
        
        processing_time = (time.time() - start_time) * 1000
        performance_metrics.record_request(time.time() - start_time)
        
        return {
            "message": "Issue deleted successfully",
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete issue {issue_id}: {e}", exc_info=True)
        performance_metrics.record_request(time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete issue: {str(e)}")

# ========================================
# MONITORING AND HEALTH ENDPOINTS
# ========================================

@router.get("/issues/health")
async def health_check():
    """
    🏥 Health check endpoint
    """
    try:
        # Check Redis connection
        redis_service = await get_redis_cluster_service()
        redis_status = "healthy" if redis_service else "unhealthy"
        
        # Check MongoDB connection
        mongodb_service = await get_optimized_mongodb_service()
        mongodb_status = "healthy" if mongodb_service else "unhealthy"
        
        # Overall status
        overall_status = "healthy" if redis_status == "healthy" and mongodb_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "redis": redis_status,
                "mongodb": mongodb_status
            },
            "performance_metrics": performance_metrics.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/issues/metrics")
async def get_performance_metrics():
    """
    📈 Get detailed performance metrics
    """
    try:
        redis_service = await get_redis_cluster_service()
        redis_stats = await redis_service.get_stats() if redis_service else {}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "application_metrics": performance_metrics.get_stats(),
            "redis_metrics": redis_stats,
            "rate_limiter_metrics": rate_limiter.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

logger.info("🚀 Ultra-optimized Issues Router loaded successfully - Ready for 1 Lakh+ users!")
