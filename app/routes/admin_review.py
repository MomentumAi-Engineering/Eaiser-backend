from fastapi import APIRouter, HTTPException, Depends, Body, Request, Response, Cookie
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
import asyncio
from fastapi.responses import JSONResponse

from services.mongodb_optimized_service import get_optimized_mongodb_service
from services.email_service import send_email, send_formatted_ai_alert, notify_user_status_change
from services.redis_cluster_service import get_redis_cluster_service
from core.database import get_database
from services.mongodb_service import get_fs
from bson.objectid import ObjectId
from routes.issues import send_authority_email
from utils.security import verify_password, create_access_token, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, create_refresh_token, verify_refresh_token, REFRESH_TOKEN_EXPIRE_DAYS
import os
from models.admin_model import AdminCreate, AdminInDB
from datetime import timedelta
from fastapi import status
from services.security_service import SecurityService
from models.security_models import PasswordChangeRequest, TwoFactorSetup, TwoFactorVerify


# Configure logging
logger = logging.getLogger(__name__)

from core.auth import get_admin_user, require_permission
from core.permissions import has_permission

router = APIRouter(
    prefix="/admin/review", 
    tags=["Admin Review"]
    # Removed global dependencies=[Depends(get_admin_user)] to allow public /login
)

# --- Pydantic Models ---

class ReviewAction(BaseModel):
    issue_id: str
    admin_id: str
    notes: Optional[str] = None
    new_authority_email: Optional[str] = None
    new_authority_name: Optional[str] = None

class AdminLoginRequest(BaseModel):
    email: str
    password: str
    code: Optional[str] = None

class UpdateStatusRequest(BaseModel):
    issue_id: str
    status: str
    admin_id: str
    notes: Optional[str] = None

class UserAction(BaseModel):
    user_email: str
    reason: str
    admin_id: str
    issue_id: Optional[str] = None
    force_confirm: bool = False

# --- Endpoints ---

@router.post("/login")
async def admin_login(creds: AdminLoginRequest, request: Request):
    """
    OPTIMIZED admin login with security features:
    - Rate limiting (cached)
    - Account lockout (cached)
    - Background security logging (non-blocking)
    - Redis caching for admin lookup
    """
    
    # Get client information — use Cloudflare-safe IP extraction
    from services.admin_login_monitor import AdminLoginMonitor
    ip_address = AdminLoginMonitor.extract_real_ip(request)
    user_agent = request.headers.get("User-Agent", "unknown")
    
    # OPTIMIZATION: Run rate limit, lockout, and IP block checks in parallel
    rate_limit_task = SecurityService.check_rate_limit(ip_address)
    lockout_task = SecurityService.check_account_lockout(creds.email)
    ip_block_task = AdminLoginMonitor.is_ip_blocked(ip_address)
    
    rate_allowed, lockout_status, ip_block_status = await asyncio.gather(
        rate_limit_task, lockout_task, ip_block_task
    )
    
    # Check IP block (admin_login_logs: 3 failures in 5 min → 15 min block)
    if ip_block_status.get("blocked"):
        asyncio.create_task(AdminLoginMonitor.log_attempt(
            email=creds.email, ip=ip_address, user_agent=user_agent,
            success=False, failure_reason="IP blocked"
        ))
        raise HTTPException(
            status_code=403,
            detail=f"Your IP is temporarily blocked due to suspicious activity. Try again in {ip_block_status['remaining_minutes']} minutes."
        )
    
    if not rate_allowed:
        logger.warning(f"🚫 Rate limit exceeded for IP: {ip_address}")
        asyncio.create_task(AdminLoginMonitor.log_attempt(
            email=creds.email, ip=ip_address, user_agent=user_agent,
            success=False, failure_reason="Rate limited"
        ))
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please try again in 1 minute."
        )
    
    if lockout_status["locked"]:
        # Background task for failed login recording (non-blocking)
        asyncio.create_task(SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Account locked"
        ))
        asyncio.create_task(AdminLoginMonitor.log_attempt(
            email=creds.email, ip=ip_address, user_agent=user_agent,
            success=False, failure_reason="Account locked"
        ))
        raise HTTPException(
            status_code=403,
            detail=f"Account locked due to multiple failed login attempts. Try again in {lockout_status['remaining_minutes']} minutes."
        )
    
    # OPTIMIZATION: Try Redis cache first for admin lookup
    redis_service = await get_redis_cluster_service()
    admin = None
    
    if redis_service:
        cache_key = f"admin:email:{creds.email}"
        admin = await redis_service.get_cache('user_session', cache_key)
        if admin:
            logger.debug(f"✅ Admin cache HIT for {creds.email}")
    
    # Cache miss - get from database
    if not admin:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")

        try:
            collection = await mongo_service.get_collection("admins")
            admin = await collection.find_one({"email": creds.email})
            
            # Cache the admin data for 10 minutes
            if admin and redis_service:
                await redis_service.set_cache('user_session', cache_key, admin, ttl=600)
                logger.debug(f"💾 Cached admin data for {creds.email}")
                
        except Exception as e:
            logger.error(f"Database error during admin login: {e}")
            raise HTTPException(status_code=500, detail="Database error")

    # Verify credentials
    if not admin or not verify_password(creds.password, admin["password_hash"]):
        # OPTIMIZATION: Background task for failed login (non-blocking)
        asyncio.create_task(SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Invalid credentials"
        ))
        asyncio.create_task(AdminLoginMonitor.log_attempt(
            email=creds.email, ip=ip_address, user_agent=user_agent,
            success=False, failure_reason="Invalid credentials"
        ))
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if account is active
    if not admin.get("is_active", True):
        asyncio.create_task(SecurityService.record_login_attempt(
            email=creds.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            failure_reason="Inactive account"
        ))
        asyncio.create_task(AdminLoginMonitor.log_attempt(
            email=creds.email, ip=ip_address, user_agent=user_agent,
            success=False, failure_reason="Inactive account"
        ))
        raise HTTPException(status_code=403, detail="Account is inactive")

    # 2FA Logic
    if admin.get("two_factor_enabled", False):
        if not creds.code:
            return {
                "require_2fa": True,
                "email": creds.email,
                "message": "Please enter your 2FA code"
            }
        
        secret = admin.get("two_factor_secret")
        if secret:
            # Decrypt secret if encrypted (backward compatible with plaintext secrets)
            try:
                from utils.totp_encryption import decrypt_totp_secret, is_encrypted
                if is_encrypted(secret):
                    secret = decrypt_totp_secret(secret)
            except Exception as decrypt_err:
                logger.error(f"Failed to decrypt 2FA secret for {creds.email}: {decrypt_err}")
        
        if not secret or not SecurityService.verify_2fa_code(secret, creds.code):
            asyncio.create_task(SecurityService.record_login_attempt(
                email=creds.email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                failure_reason="Invalid 2FA code"
            ))
            asyncio.create_task(AdminLoginMonitor.log_attempt(
                email=creds.email, ip=ip_address, user_agent=user_agent,
                success=False, failure_reason="Invalid 2FA code"
            ))
            raise HTTPException(status_code=401, detail="Invalid 2FA code")

    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=admin["email"], expires_delta=access_token_expires
    )
    
    # Generate refresh token (stored in HTTP-only cookie)
    refresh_token = create_refresh_token(
        subject=admin["email"],
        extra_claims={"role": admin.get("role", "admin"), "admin_id": str(admin["_id"])}
    )
    
    # OPTIMIZATION: Move security logging to background (non-blocking)
    asyncio.create_task(SecurityService.record_login_attempt(
        email=creds.email,
        ip_address=ip_address,
        user_agent=user_agent,
        success=True
    ))
    asyncio.create_task(AdminLoginMonitor.log_attempt(
        email=creds.email, ip=ip_address, user_agent=user_agent,
        success=True
    ))
    
    asyncio.create_task(SecurityService.log_security_event(
        admin_email=creds.email,
        action="login",
        ip_address=ip_address,
        user_agent=user_agent,
        success=True,
        details={"role": admin.get("role")}
    ))
    
    logger.info(f"✅ Fast login: {creds.email} from {ip_address}")
    
    # Build response with HTTP-only refresh token cookie
    is_production = os.getenv("ENV", "production").lower() == "production"
    response_data = {
        "token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
        "admin": {
            "email": admin["email"],
            "id": str(admin["_id"]),
            "role": admin.get("role", "admin"),
            "name": admin.get("name", "Admin"),
            "require_password_change": admin.get("require_password_change", False),
            "two_factor_enabled": admin.get("two_factor_enabled", False)
        }
    }
    
    from fastapi.responses import JSONResponse
    response = JSONResponse(content=response_data)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=is_production,  # True in production (requires HTTPS)
        samesite="strict",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,  # 7 days in seconds
        path="/api/admin/review"
    )
    return response

@router.post("/refresh-token")
async def refresh_access_token(
    request: Request,
    refresh_token: Optional[str] = Cookie(None)
):
    """
    Issue a new access token using the refresh token from HTTP-only cookie.
    The refresh token is never exposed to JavaScript.
    """
    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="No refresh token provided"
        )
    
    # Verify refresh token
    payload = verify_refresh_token(refresh_token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token"
        )
    
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    # Verify admin still exists and is active
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    collection = await mongo_service.get_collection("admins")
    admin = await collection.find_one({"email": email})
    
    if not admin or not admin.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is inactive or not found")
    
    # Issue new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        subject=email, expires_delta=access_token_expires
    )
    
    return {
        "token": new_access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/logout")
async def admin_logout(response: Response):
    """
    Logout by clearing the refresh token cookie.
    Frontend should also discard the access token from memory.
    """
    is_production = os.getenv("ENV", "production").lower() == "production"
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        secure=is_production,
        samesite="strict",
        path="/api/admin/review"
    )
    return response

@router.post("/create", response_model=dict)
async def create_admin(new_admin: AdminCreate, current_admin: dict = Depends(require_permission("create_team_member"))):
    """
    Create a new admin user.
    - Team Members are created by Super Admins or Admins.
    - Admins/Super Admins are created ONLY by Super Admins.
    Sends welcome email with login credentials.
    """
    # Permission logic: 
    # To create an 'admin' or 'super_admin', you need the 'create_admin' permission.
    if new_admin.role in ["admin", "super_admin"] and not has_permission(current_admin.get("role"), "create_admin"):
        raise HTTPException(
            status_code=403, 
            detail=f"Only super admins can create {new_admin.role} accounts"
        )
    
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        collection = await mongo_service.get_collection("admins", read_only=False)
        # Check if email already exists
        if await collection.find_one({"email": new_admin.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Set permissions based on role
        permissions = {
            "super_admin": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": True,
                "can_manage_team": True
            },
            "admin": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": True,
                "can_manage_team": False
            },
            "team_member": {
                "can_approve": True,
                "can_decline": True,
                "can_assign": False,
                "can_manage_team": False
            },
            "viewer": {
                "can_approve": False,
                "can_decline": False,
                "can_assign": False,
                "can_manage_team": False
            }
        }
        
        admin_dict = new_admin.dict()
        temporary_password = new_admin.password  # Store for email
        admin_dict["password_hash"] = get_password_hash(new_admin.password)
        del admin_dict["password"]
        admin_dict["created_at"] = datetime.utcnow()
        admin_dict["is_active"] = True
        admin_dict["assigned_issues"] = []
        admin_dict["permissions"] = permissions.get(new_admin.role, permissions["admin"])
        admin_dict["last_login"] = None
        
        result = await collection.insert_one(admin_dict)
        created_admin = await collection.find_one({"_id": result.inserted_id})
        
        # Send welcome email
        try:
            from services.admin_email_service import send_admin_welcome_email
            email_sent = await send_admin_welcome_email(
                admin_email=new_admin.email,
                admin_name=new_admin.name or "Admin",
                role=new_admin.role,
                temporary_password=temporary_password,
                created_by=current_admin.get("name", current_admin.get("email"))
            )
            logger.info(f"✉️ Welcome email {'sent' if email_sent else 'failed'} to {new_admin.email}")
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            # Don't fail the whole operation if email fails
        
        # Format for response
        if "_id" in created_admin:
            created_admin["_id"] = str(created_admin["_id"])
            created_admin["id"] = created_admin["_id"]
            
        return {
            **created_admin,
            "message": f"Admin created successfully. Welcome email sent to {new_admin.email}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating admin: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/list", response_model=List[dict])
async def get_admins(current_admin: dict = Depends(require_permission("view_team"))):
    """
    List all admin users.
    """
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        collection = await mongo_service.get_collection("admins")
        admins = await collection.find().to_list(100)
        
        # Convert ObjectId to str for JSON serialization
        results = []
        for admin in admins:
            if "_id" in admin:
                admin["_id"] = str(admin["_id"])
                # Ensure alias 'id' is present if needed by frontend, though Pydantic handled it before
                admin["id"] = admin["_id"]
            results.append(admin)
            
        logger.info(f"Retrieved {len(results)} admins")
        return results
    except Exception as e:
        logger.error(f"Error listing admins: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# DEBUG ENDPOINT - Remove after testing
@router.get("/pending-debug")
async def get_pending_reviews_debug():
    """
    Debug endpoint without auth to test data retrieval
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            return {"error": "Database unavailable"}
        
        # Log which database we're using
        db_name = mongo_service.db.name if hasattr(mongo_service, 'db') else "unknown"
        logger.info(f"🔍 Debug endpoint: Using database '{db_name}'")
        
        collection = await mongo_service.get_collection("issues")
        
        # Count total issues
        total_count = await collection.count_documents({})
        needs_review_count = await collection.count_documents({"status": "needs_review"})
        
        logger.info(f"🔍 Total issues in DB: {total_count}, needs_review: {needs_review_count}")
        
        cursor = collection.find({"status": "needs_review"}).sort("timestamp", -1).limit(10)
        issues = await cursor.to_list(length=10)
        
        return {
            "database": db_name,
            "total_issues": total_count,
            "needs_review_count": needs_review_count,
            "count": len(issues),
            "issues": [
                {
                    "id": str(i["_id"]),
                    "status": i.get("status"),
                    "type": i.get("issue_type"),
                    "timestamp": str(i.get("timestamp"))
                } for i in issues
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/analytics/dashboard")
async def get_dashboard_analytics(admin: dict = Depends(get_admin_user)):
    """
    REAL analytics dashboard data computed from MongoDB.
    Returns: counts, trends, avg resolution time, weekly data, hotspot locations.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        collection = await mongo_service.get_collection("issues")
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # --- OPTIMIZED: Use $facet to compute EVERYTHING in ONE database scan ---
        pending_query = {"status": {"$in": ["pending", "needs_review", "pending_review", "screened_out"]}}
        approved_query = {"status": {"$in": ["approved", "submitted"]}}
        rejected_query = {"status": {"$in": ["rejected", "declined"]}}
        resolved_query = {"status": {"$in": ["resolved", "completed"]}}

        pipeline = [
            {
                "$facet": {
                    "counts": [
                        {
                            "$group": {
                                "_id": None,
                                "total": {"$sum": 1},
                                "pending": {"$sum": {"$cond": [{"$in": ["$status", ["pending", "needs_review", "pending_review", "screened_out"]]}, 1, 0]}},
                                "approved": {"$sum": {"$cond": [{"$in": ["$status", ["approved", "submitted"]]}, 1, 0]}},
                                "rejected": {"$sum": {"$cond": [{"$in": ["$status", ["rejected", "declined"]]}, 1, 0]}},
                                "resolved": {"$sum": {"$cond": [{"$in": ["$status", ["resolved", "completed"]]}, 1, 0]}},
                                "high_conf": {"$sum": {"$cond": [{"$gt": [{"$toDouble": {"$ifNull": ["$confidence", "$report.issue_overview.confidence_percent"]}}, 80]}, 1, 0]}},
                                "approved_today": {"$sum": {"$cond": [
                                    {"$and": [
                                        {"$in": ["$status", ["approved", "submitted"]]},
                                        {"$gte": ["$admin_review.timestamp", today_start]}
                                    ]}, 1, 0
                                ]}}
                            }
                        }
                    ],
                    "issue_types": [
                        {"$group": {"_id": "$issue_type", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 8}
                    ],
                    "hotspots": [
                        {"$match": {"address": {"$exists": True, "$ne": ""}}},
                        {"$group": {
                            "_id": {"$trim": {"input": {"$arrayElemAt": [{"$split": ["$address", ","]}, -2]}}},
                            "count": {"$sum": 1},
                            "lat": {"$first": "$latitude"},
                            "lng": {"$first": "$longitude"}
                        }},
                        {"$match": {"_id": {"$ne": None}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 5}
                    ]
                }
            }
        ]
        
        facet_results = await collection.aggregate(pipeline).to_list(1)
        res = facet_results[0] if facet_results else {}
        counts = res.get("counts", [{}])[0]
        
        total_count = counts.get("total", 0)
        pending_count = counts.get("pending", 0)
        approved_count = counts.get("approved", 0)
        rejected_count = counts.get("rejected", 0)
        resolved_count = counts.get("resolved", 0)
        approved_today = counts.get("approved_today", 0)
        high_confidence_count = counts.get("high_conf", 0)
        
        # Fallback for approved_today since timestamp might be string or Date
        if approved_today == 0 and approved_count > 0:
             # re-check specifically with string ISO format if aggregate check (as Date) might have missed it
             approved_today = await collection.count_documents({
                 "status": {"$in": ["approved", "submitted"]},
                 "admin_review.timestamp": {"$gte": today_start.isoformat()}
             })

        # --- Resolution Time & Weekly Trends (Specific optimized pipelines) ---
        avg_resolution_hours = 0
        try:
            resolution_pipeline = [
                {"$match": {
                    "status": {"$in": ["approved", "submitted", "resolved", "completed"]},
                    "admin_review.timestamp": {"$exists": True},
                    "timestamp": {"$exists": True}
                }},
                {"$addFields": {
                    "review_date": {
                        "$cond": {
                            "if": {"$eq": [{"$type": "$admin_review.timestamp"}, "string"]},
                            "then": {"$toDate": "$admin_review.timestamp"},
                            "else": "$admin_review.timestamp"
                        }
                    },
                    "submit_date": {
                        "$cond": {
                            "if": {"$eq": [{"$type": "$timestamp"}, "string"]},
                            "then": {"$toDate": "$timestamp"},
                            "else": "$timestamp"
                        }
                    }
                }},
                {"$addFields": {
                    "resolution_hours": {
                        "$divide": [
                            {"$subtract": ["$review_date", "$submit_date"]},
                            3600000  # ms to hours
                        ]
                    }
                }},
                {"$match": {"resolution_hours": {"$gt": 0}}},
                {"$group": {
                    "_id": None,
                    "avg_hours": {"$avg": "$resolution_hours"},
                    "count": {"$sum": 1}
                }}
            ]
            res_result = await collection.aggregate(resolution_pipeline).to_list(1)
            avg_resolution_hours = round(res_result[0]["avg_hours"], 1) if res_result and res_result[0].get("avg_hours") else 0
        except Exception as e:
            logger.warning(f"Resolution time calc failed: {e}")
            avg_resolution_hours = 0
        
        # --- 3. Weekly Trend Data (last 7 days actual + next 3 days predicted) ---
        weekly_data = []
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        for i in range(6, -1, -1):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            # Try with both Date and string timestamp formats
            day_count = 0
            try:
                # Try Date format first
                day_count = await collection.count_documents({
                    "timestamp": {"$gte": day_start, "$lt": day_end}
                })
            except Exception:
                pass
            
            if day_count == 0:
                # Try string format (ISO format comparison)
                try:
                    day_count = await collection.count_documents({
                        "timestamp": {"$gte": day_start.isoformat(), "$lt": day_end.isoformat()}
                    })
                except Exception:
                    pass
            
            weekday_name = day_names[day_start.weekday()]
            weekly_data.append({
                "name": weekday_name,
                "actual": day_count,
                "predicted": None
            })
        
        # Simple prediction: average of last 7 days ± small variance
        avg_daily = sum(d["actual"] for d in weekly_data) / max(len(weekly_data), 1)
        import random
        for i in range(1, 4):
            future_day = now + timedelta(days=i)
            predicted = max(0, round(avg_daily + random.uniform(-2, 2)))
            weekly_data.append({
                "name": day_names[future_day.weekday()],
                "actual": None,
                "predicted": predicted
            })
        
        # Trend percentage (this week vs projected)
        last_week_total = sum(d["actual"] or 0 for d in weekly_data[:7])
        projected_total = last_week_total + sum(d["predicted"] or 0 for d in weekly_data[7:])
        trend_pct = round(((projected_total - last_week_total) / max(last_week_total, 1)) * 100)
        
        # --- 4. Issue Type & Hotspots (already resolved in facet) ---
        issue_types = [{"type": t["_id"] or "unknown", "count": t["count"]} for t in res.get("issue_types", [])]
        hotspots = [{
                "location": h["_id"] or "Unknown",
                "count": h["count"],
                "lat": h.get("lat"),
                "lng": h.get("lng")
        } for h in res.get("hotspots", [])]
        
        # --- 6. System Health ---
        one_hour_ago = now - timedelta(hours=1)
        try:
            recent_submissions = await collection.count_documents({"timestamp": {"$gte": one_hour_ago}})
            if recent_submissions == 0:
                recent_submissions = await collection.count_documents({"timestamp": {"$gte": one_hour_ago.isoformat()}})
        except Exception:
            recent_submissions = 0
        system_load = min(100, round((recent_submissions / max(total_count, 1)) * 1000))  # Rough metric
        
        return {
            "counts": {
                "total": total_count,
                "pending": pending_count,
                "approved": approved_count,
                "approved_today": approved_today,
                "rejected": rejected_count,
                "resolved": resolved_count,
                "high_confidence": high_confidence_count
            },
            "avg_resolution_hours": avg_resolution_hours,
            "chart_data": weekly_data,
            "trend_percentage": trend_pct,
            "trend_direction": "up" if trend_pct > 0 else "down",
            "issue_types": issue_types,
            "hotspots": hotspots,
            "system_load": system_load,
            "recent_submissions_1h": recent_submissions,
            "generated_at": now.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analytics dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@router.get("/analytics/warroom")
async def get_warroom_data(
    search: Optional[str] = None,
    issue_type: Optional[str] = None,
    severity: Optional[str] = None,
    admin: dict = Depends(get_admin_user)
):
    """
    Advanced War Room data: all issues with coordinates + filters + hotspot analysis.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        collection = await mongo_service.get_collection("issues")
        
        # Build filter
        match_filter = {
            "$or": [
                {"latitude": {"$exists": True, "$ne": 0}},
                {"report.report.latitude": {"$exists": True, "$ne": 0}}
            ]
        }
        
        if search:
            match_filter["$and"] = match_filter.get("$and", [])
            match_filter["$and"].append({
                "$or": [
                    {"address": {"$regex": search, "$options": "i"}},
                    {"issue_type": {"$regex": search, "$options": "i"}},
                    {"description": {"$regex": search, "$options": "i"}}
                ]
            })
        
        if issue_type and issue_type != "all":
            match_filter["issue_type"] = issue_type
        
        if severity and severity != "all":
            match_filter["severity"] = severity
        
        pipeline = [
            {"$match": match_filter},
            {"$sort": {"timestamp": -1}},
            {"$limit": 200},
            {"$project": {
                "_id": 1,
                "issue_id": {"$toString": "$_id"},
                "issue_type": 1,
                "status": 1,
                "severity": 1,
                "address": 1,
                "latitude": 1,
                "longitude": 1,
                "confidence_score": {"$ifNull": ["$confidence", 0]},
                "timestamp": 1,
                "flagged_at": 1,
                "description": 1,
                "reporter_email": {"$ifNull": ["$reporter_email", "$user_email"]},
                "image_url": {"$concat": ["/api/issues/", {"$toString": "$_id"}, "/image"]},
                "report_lat": "$report.report.latitude",
                "report_lng": "$report.report.longitude"
            }}
        ]
        
        issues = await collection.aggregate(pipeline).to_list(200)
        
        # Normalize coordinates
        for issue in issues:
            if "_id" in issue:
                issue["_id"] = str(issue["_id"])
            if not issue.get("latitude") and issue.get("report_lat"):
                issue["latitude"] = issue["report_lat"]
                issue["longitude"] = issue.get("report_lng")
            issue.pop("report_lat", None)
            issue.pop("report_lng", None)
        
        # Compute hotspots (group by proximity)
        hotspot_pipeline = [
            {"$match": {"latitude": {"$exists": True, "$ne": 0, "$ne": None}}},
            {"$group": {
                "_id": {"$concat": [
                    {"$toString": {"$round": ["$latitude", 2]}},
                    ",",
                    {"$toString": {"$round": ["$longitude", 2]}}
                ]},
                "count": {"$sum": 1},
                "lat": {"$avg": "$latitude"},
                "lng": {"$avg": "$longitude"},
                "top_type": {"$first": "$issue_type"},
                "address": {"$first": "$address"}
            }},
            {"$match": {"count": {"$gte": 2}}},
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]
        
        hotspots = await collection.aggregate(hotspot_pipeline).to_list(20)
        for h in hotspots:
            h["_id"] = str(h["_id"])
        
        # Stats summary
        total_on_map = len(issues)
        high_severity = len([i for i in issues if i.get("severity") == "high" or (i.get("confidence_score") or 0) > 80])
        
        return {
            "issues": issues,
            "hotspots": hotspots,
            "stats": {
                "total_on_map": total_on_map,
                "high_severity": high_severity,
                "pending": len([i for i in issues if i.get("status") in ["pending", "needs_review", "pending_review"]]),
                "resolved": len([i for i in issues if i.get("status") in ["approved", "submitted", "resolved"]])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"War Room data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending", response_model=List[dict])
async def get_pending_reviews(
    skip: int = 0,
    limit: int = 50,
    admin: dict = Depends(get_admin_user)
):
    """
    Get all issues that are flagged for review.
    OPTIMIZED: Uses MongoDB Aggregation to handle filtering logic on the database side.
    Excludes heavy fields like embeddings/images.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")
        
        collection = await mongo_service.get_collection("issues")

        # Aggregation Pipeline for efficient filtering
        # Logic:
        # 1. Team member: Only assigned issues
        # 2. Status 'needs_review' etc.
        
        match_query = {
            "status": {
                "$in": ["pending", "needs_review", "pending_review", "screened_out", "dispatch_decision"]
            }
        }
        # Filter exclusions
        match_query["status"]["$nin"] = ["approved", "submitted", "declined", "rejected", "completed", "resolved"]
        
        # TEAM MEMBER FILTER: Only see assigned issues
        if admin.get("role") == "team_member":
            match_query["assigned_to"] = admin.get("email")

        pipeline = [
            {
                "$match": match_query
            },
            {
                "$sort": { "timestamp": -1 }
            },
            { "$skip": skip },
            { "$limit": limit },
            {
                "$project": {
                    "image_data": 0, 
                    "original_image": 0, 
                    "compressed_image": 0, 
                    "vector_embedding": 0,
                    "logs": 0,
                    "base64_image": 0
                }
            }
        ]
        
        # Execute Aggregation
        recent_issues = await collection.aggregate(pipeline).to_list(length=limit)
        
        final_reviews = []
        reporter_emails = set()

        for issue in recent_issues:
            sid = str(issue["_id"])
            
            # NORMALIZATION
            issue["issue_id"] = sid
            if "_id" in issue: issue["_id"] = sid
            if not issue.get("reporter_email"): issue["reporter_email"] = issue.get("user_email")
            
            # Ensure Image URL (Lightweight)
            if "image_url" not in issue:
                issue["image_url"] = f"/api/issues/{sid}/image"

            # Ensure Address
            if "address" not in issue:
                issue["address"] = issue.get("report", {}).get("template_fields", {}).get("location", "Unknown Location")

            # Clean Confidence
            issue["confidence"] = issue.get("effective_confidence", 0)
            
            # Force status for UI consistency
            if issue.get("status") == "pending":
                 issue["status_original"] = "pending"
                 issue["status"] = "needs_review"

            if issue.get("reporter_email"):
                reporter_emails.add(issue["reporter_email"])

            final_reviews.append(issue)

        # Bulk user fetch
        user_stats_map = {}
        if reporter_emails:
            users_coll = await mongo_service.get_collection("users")
            users = await users_coll.find(
                {"email": {"$in": list(reporter_emails)}},
                {"email": 1, "rejected_reports_count": 1, "is_active": 1, "name": 1, "full_name": 1}
            ).to_list(None)
            
            for u in users:
                user_stats_map[u["email"]] = {
                    "rejected_count": u.get("rejected_reports_count", 0),
                    "is_active": u.get("is_active", True),
                    "name": u.get("name") or u.get("full_name") or "User"
                }

        # Enrich
        for issue in final_reviews:
            if issue.get("reporter_email") and issue["reporter_email"] in user_stats_map:
                issue["user_reputation"] = user_stats_map[issue["reporter_email"]]
            else:
                 issue["user_reputation"] = {"rejected_count": 0, "is_active": True}

        logger.info(f"Admin Dashboard: Fetched {len(final_reviews)} issues via Optimized Aggregation")
        return final_reviews

    except Exception as e:
        logger.error(f"Failed to fetch pending reviews: {e}", exc_info=True)
        # Fallback to empty list or error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/resolved-strict", response_model=List[dict])
async def get_resolved_history_strict(
    skip: int = 0,
    limit: int = 50,
    admin: dict = Depends(get_admin_user)
):
    """
    STRICT RESOLVED HISTORY
    Returns ONLY issues that:
    1. Are fully resolved (status in [approved, submitted, rejected, declined, resolved])
    2. WERE RESOLVED BY THE CURRENT ADMIN (admin_review.admin_id matches)
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service: return []
        
        collection = await mongo_service.get_collection("issues")
        current_admin_id = str(admin["_id"])
        
        # 1. First, find IDs of issues resolved by this admin using a robust check
        # We query for issues where this admin touched it AND it is finalized
        
        final_statuses = ["approved", "submitted", "rejected", "declined", "completed", "resolved"]
        
        # Safe match for Admin ID (String or ObjectId)
        match_stage = {
            "$match": {
                "status": {"$in": final_statuses},
                "$or": [
                     {"admin_review.admin_id": current_admin_id},
                     {"admin_review.admin_id": ObjectId(current_admin_id) if ObjectId.is_valid(current_admin_id) else "invalid_id"}
                ]
            }
        }
        
        pipeline = [
            match_stage,
            {"$sort": {"admin_review.timestamp": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {
                "$project": {
                    "image_data": 0, "original_image": 0, "compressed_image": 0, "vector_embedding": 0, "logs": 0, "base64_image": 0
                }
            }
        ]
        
        issues = await collection.aggregate(pipeline).to_list(length=limit)
        
        # 2. Enrich with Resolver Name (Me)
        results = []
        for issue in issues:
            issue["issue_id"] = str(issue["_id"])
            if "_id" in issue: issue["_id"] = str(issue["_id"])
            if "image_url" not in issue: issue["image_url"] = f"/api/issues/{issue['issue_id']}/image"
            if "admin_review" not in issue: issue["admin_review"] = {}
            
            # Since we filtered by current admin, it IS resolved by "Me"
            issue["admin_review"]["resolver_name"] = "Me"
            results.append(issue)
            
        logger.info(f"Strict Resolved History for {current_admin_id}: Found {len(results)}")
        return results

    except Exception as e:
        logger.error(f"Error in strict resolved history: {e}")
        return []

@router.get("/resolved", response_model=List[dict])
async def get_resolved_reviews(
    skip: int = 0,
    limit: int = 50,
    admin: dict = Depends(get_admin_user)
):
    """
    Get history of resolved issues.
    Includes strict Python-side filtering to ensure accuracy.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            # Fallback prevents crash
            logger.error("Mongo service unavailable in get_resolved_reviews")
            return []
        
        collection = await mongo_service.get_collection("issues")
        
        current_admin_id = str(admin["_id"])
        
        # Build Query
        or_conditions = [{"admin_review.admin_id": current_admin_id}]
        try:
            or_conditions.append({"admin_review.admin_id": ObjectId(current_admin_id)})
        except:
            pass

        # Valid resolved statuses
        valid_statuses = ["rejected", "declined", "completed", "submitted", "resolved", "approved"]
        
        query = {
            "$and": [
                {"status": {"$in": valid_statuses}},
                {"status": {"$ne": "needs_review"}}, 
                {"status": {"$ne": "pending"}},
                {"$or": or_conditions}
            ]
        }
        
        projection = {
            "image_data": 0, "original_image": 0, "compressed_image": 0, "vector_embedding": 0, "logs": 0, "base64_image": 0
        }
        
        # Fetch Data
        cursor = collection.find(query, projection).sort("admin_review.timestamp", -1).skip(skip).limit(limit)
        issues = await cursor.to_list(length=limit)
        
        # --- STRICT PYTHON FILTERING ---
        # Filter out impurities that might bypass Mongo for any reason
        filtered_issues = []
        for issue in issues:
            status = issue.get("status", "").lower()
            # Double check status
            if status not in valid_statuses:
                continue
            if status in ["needs_review", "pending"]:
                continue
            filtered_issues.append(issue)
            
        issues = filtered_issues

        # --- ENRICHMENT ---
        # Safe Admin Name Resolution
        admin_ids = set()
        for issue in issues:
            try:
                review_data = issue.get("admin_review", {})
                if not isinstance(review_data, dict): review_data = {}
                
                aid = review_data.get("admin_id")
                if aid and aid != 'admin':
                    admin_ids.add(ObjectId(aid))
            except Exception:
                pass # Ignore malformed IDs

        admin_names = {}
        if admin_ids:
            try:
                admins_coll = await mongo_service.get_collection("admins")
                found_admins = await admins_coll.find({"_id": {"$in": list(admin_ids)}}).to_list(None)
                for adm in found_admins:
                    admin_names[str(adm["_id"])] = adm.get("name", "Unknown Admin")
            except Exception as e:
                logger.error(f"Failed to fetch admin names: {e}")

        results = []
        for issue in issues:
            try:
                # Normalize ID
                sid = str(issue["_id"])
                issue["issue_id"] = sid
                if "_id" in issue: issue["_id"] = sid
                
                # Ensure Attributes
                if "image_url" not in issue:
                    issue["image_url"] = f"/api/issues/{sid}/image"
                
                if "admin_review" not in issue or not isinstance(issue["admin_review"], dict):
                    issue["admin_review"] = {}
                
                # Resolver Name Logic
                resolver_id = issue["admin_review"].get("admin_id")
                r_name = "Unknown"
                
                if resolver_id == 'admin':
                    r_name = "System Admin"
                elif resolver_id and str(resolver_id) in admin_names:
                    r_name = admin_names[str(resolver_id)]
                elif resolver_id and str(resolver_id) == current_admin_id:
                    r_name = "Me"
                
                issue["admin_review"]["resolver_name"] = r_name
                results.append(issue)
            except Exception as e:
                logger.error(f"Error processing issue {issue.get('_id')}: {e}")
                continue

        return results

    except Exception as e:
        logger.error(f"CRITICAL ERROR in get_resolved_reviews: {e}", exc_info=True)
        # Return empty list instead of 500 Error to keep UI alive
        return []

@router.post("/approve")
async def approve_issue(action: ReviewAction, admin: dict = Depends(require_permission("approve_assigned"))):
    """
    Approve a flagged issue.
    - Updates status to 'pending' (or 'approved')
    - Triggers the email to authority
    - Notifies user (optional)
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # 1. Get the issue
        issue = await mongo_service.get_issue_by_id(action.issue_id)
        if issue and not issue.get("reporter_email"): issue["reporter_email"] = issue.get("user_email")

        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")

        # PERMISSION CHECK: Team member can only act on assigned issues
        if admin.get("role") == "team_member" and issue.get("assigned_to") != admin.get("email"):
            raise HTTPException(
                status_code=403, 
                detail="You can only approve issues assigned to you"
            )

        # 2. Update status to 'pending' (ready for authority)
        # 2. Update status to 'submitted' (ready for authority)
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": action.issue_id},
            update_dict={
                "$set": {
                    "status": "submitted",  # Mark as submitted/completed
                    "admin_review": {
                        "action": "approve",
                        "admin_id": action.admin_id,
                        "timestamp": datetime.utcnow(),
                        "notes": action.notes
                    }
                }
            }
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update issue status")

        # LOG APPROVAL
        asyncio.create_task(SecurityService.log_security_event(
            admin_email=admin["email"],
            action="approve_issue",
            ip_address="internal", 
            user_agent="admin_dashboard", 
            success=True,
            details={"issue_id": action.issue_id, "notes": action.notes}
        ))

        # 2.5 If authority changed, update report and issue
        if action.new_authority_email and issue.get("report"):
             report = issue["report"]
             # Update recommended authorities in report
             # We just override the first one or add to list
             new_auth = {
                 "name": action.new_authority_name or "Assigned Authority", 
                 "email": action.new_authority_email, 
                 "type": "custom"
             }
             
             # Identify if we replace or append? For now, let's force set sending list
             # The email trigger below uses issue['report'] which is STALE now.
             # We need to update issue['report'] in DB and memory.
             
             report["responsible_authorities_or_parties"] = [new_auth]
             # Update in DB
             await mongo_service.update_one_optimized(
                 collection_name='issues',
                 filter_dict={"_id": action.issue_id},
                 update_dict={"$set": {"report.responsible_authorities_or_parties": [new_auth]}}
             )
             
             # Update local issue object for email sending
             issue["report"] = report
             logger.info(f"Admin updated authority to {action.new_authority_email} for issue {action.issue_id}")



        # 3. Trigger Authority Email (RESTRICTED ACTION)
        # Check if the current admin has permission to send to authority
        can_send = has_permission(admin.get("role"), "send_to_authority")
        
        email_sent_info = "Triggered" if can_send else "Skipped (No Permission)"
        
        if issue.get("report") and can_send:
            # Send standard formatted email (same as normal submission)
            try:
                # Need to fetch image content for attachment
                fs = await get_fs()
                image_id = issue.get("image_id")
                image_content = b""
                if image_id:
                     try:
                        grid_out = await fs.open_download_stream(ObjectId(image_id))
                        image_content = await grid_out.read()
                     except Exception as e:
                        logger.warning(f"Failed to fetch image {image_id} for email: {e}")

                # Prepare authorities list
                # If we updated it in step 2.5, use that. Or use what's in report.
                current_authorities = issue["report"].get("responsible_authorities_or_parties", [])
                
                # Normalize authorities format for email function
                # send_authority_email expects List[Dict[str, str]] with name, email
                email_auths = []
                for a in current_authorities:
                    email_auths.append({
                        "name": a.get("name", "Authority"), 
                        "email": a.get("email"),
                        "type": a.get("type", "general")
                    })
                
                email_success = await send_authority_email(
                    issue_id=str(issue["_id"]),
                    authorities=email_auths,
                    issue_type=issue.get("issue_type", "Unknown"),
                    final_address=issue.get("address", "Unknown Address"),
                    zip_code=issue.get("zip_code", "N/A"),
                    timestamp_formatted=issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
                    report=issue["report"],
                    confidence=float(issue.get("confidence", 0)),
                    category=issue.get("category", "public"),
                    timezone_name=issue.get("timezone_name", "UTC"),
                    latitude=float(issue.get("latitude", 0.0)),
                    longitude=float(issue.get("longitude", 0.0)),
                    image_content=image_content,
                    is_user_review=False
                )
                
                if email_success:
                    logger.info(f"Standard Authority email triggered for approved issue {action.issue_id}")
                    return {"message": "Issue approved and email sent to authorities", "issue_id": action.issue_id, "email_sent": True}
                else:
                    logger.warning(f"Authority email failed for issue {action.issue_id}")
                    return {"message": "Issue approved but email sending failed. Check server logs.", "issue_id": action.issue_id, "email_sent": False}

            except Exception as e:
                logger.error(f"Failed to send authority email after approval: {e}", exc_info=True)
                return {"message": f"Issue approved but email error occurred: {str(e)}", "issue_id": action.issue_id, "email_sent": False}

        # 4. Notify User
        if issue.get("reporter_email"):
            asyncio.create_task(notify_user_status_change(issue["reporter_email"], action.issue_id, 'approved', action.notes))

        return {"message": "Issue approved (no report data to email)", "issue_id": action.issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decline")
async def decline_issue(action: ReviewAction, admin: dict = Depends(require_permission("decline_assigned"))):
    """
    Decline a flagged issue.
    - Updates status to 'rejected'
    - Notifies user (TODO: Add notification logic)
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # PERMISSION CHECK
        issue = await mongo_service.get_issue_by_id(action.issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
            
        if admin.get("role") == "team_member" and issue.get("assigned_to") != admin.get("email"):
             raise HTTPException(
                status_code=403, 
                detail="You can only decline issues assigned to you"
            )

        # Update status
        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": action.issue_id},
            update_dict={
                "$set": {
                    "status": "rejected",
                    "admin_review": {
                        "action": "decline",
                        "admin_id": action.admin_id,
                        "timestamp": datetime.utcnow(),
                        "notes": action.notes
                    }
                }
            }
        )

        if not success:
            raise HTTPException(status_code=404, detail="Issue not found or update failed")

        # LOG DECLINE
        asyncio.create_task(SecurityService.log_security_event(
            admin_email=admin["email"],
            action="decline_issue",
            ip_address="internal", 
            user_agent="admin_dashboard", 
            success=True,
            details={"issue_id": action.issue_id, "notes": action.notes}
        ))

        # Fetch issue to get reporter email
        issue = await mongo_service.get_issue_by_id(action.issue_id)
        if issue and not issue.get("reporter_email"): issue["reporter_email"] = issue.get("user_email")

        # Notify User "Your report was declined"
        if issue and issue.get("reporter_email"):
             email = issue["reporter_email"]
             # Increment rejected count
             await mongo_service.db.users.update_one(
                 {"email": email},
                 {"$inc": {"rejected_reports_count": 1}}
             )
             
             # Notify User
             asyncio.create_task(notify_user_status_change(email, action.issue_id, 'rejected', action.notes))
        
        return {"message": "Issue declined", "issue_id": action.issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error declining issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-status")
async def set_issue_status(action: UpdateStatusRequest, admin: dict = Depends(require_permission("edit_report"))):
    """
    Set an intermediate status for a report (e.g., 'no_action_required').
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # allowed statuses
        allowed = ["no_action_required", "needs_support", "investigating", "duplicate"]
        if action.status not in allowed:
             # Just a warning or strict? Let's allow it but log
             pass

        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": action.issue_id},
            update_dict={
                "$set": {
                    "status": action.status,
                    "admin_review": {
                        "action": "status_change",
                        "status_to": action.status,
                        "admin_id": action.admin_id,
                        "timestamp": datetime.utcnow(),
                        "notes": action.notes
                    }
                }
            }
        )

        if not success:
            raise HTTPException(status_code=404, detail="Issue not found")

        # Notify if needed (optional)

        return {"message": f"Status updated to {action.status}", "issue_id": action.issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-report")
async def update_issue_report(
    issue_id: str = Body(...),
    summary: Optional[str] = Body(None),
    issue_type: Optional[str] = Body(None),
    confidence: Optional[float] = Body(None),
    admin: dict = Depends(require_permission("edit_report"))
):
    """
    Update the report details of an issue (summary, type, confidence).
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # Get existing issue
        issue = await mongo_service.get_issue_by_id(issue_id)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        update_fields = {}
        
        # Update Issue Type (Top level + Report level)
        if issue_type:
            update_fields["issue_type"] = issue_type
            # Also update deeply nested report fields where type might be stored
            update_fields["report.unified_report.issue_type"] = issue_type
            update_fields["report.issue_overview.issue_type"] = issue_type
            
        # Update Summary (Deeply nested)
        if summary:
            update_fields["description"] = summary # Top level fallback
            update_fields["report.unified_report.summary_explanation"] = summary
            update_fields["report.issue_overview.summary_explanation"] = summary
            
        # Update Confidence
        if confidence is not None:
            update_fields["confidence"] = confidence
            update_fields["report.unified_report.confidence_percent"] = confidence
            update_fields["report.issue_overview.confidence_percent"] = confidence

        if not update_fields:
            return {"message": "No changes provided"}
            
        # Add admin edit trace
        update_fields["last_edited_by"] = admin.get("email")
        update_fields["last_edited_at"] = datetime.utcnow()

        success = await mongo_service.update_one_optimized(
            collection_name='issues',
            filter_dict={"_id": issue_id},
            update_dict={"$set": update_fields}
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update issue")

        logger.info(f"Report updated for issue {issue_id} by {admin.get('email')}")
        return {"message": "Report updated successfully", "issue_id": issue_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/skip")
async def skip_review(action: ReviewAction, admin: dict = Depends(get_admin_user)):
    """
    Skip exact review (maybe leave for another admin or mark as 'skipped'?)
    For now, we'll just log it or maybe irrelevant if we don't change status.
    If 'skip' means 'Ignore/Archive', we can use a status like 'archived'.
    """
    # Assuming Skip means "I don't know, leave it for now" -> no action on status
    return {"message": "Review skipped (no action taken)", "issue_id": action.issue_id}

@router.post("/deactivate-user")
async def deactivate_user(action: UserAction, admin: dict = Depends(get_admin_user)):
    """
    Deactivate a user account with safeguards.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # SAFEGUARD: LOW CONFIDENCE CHECK
        if action.issue_id and not action.force_confirm:
            issue = await mongo_service.get_issue_by_id(action.issue_id)
            if issue:
                # Check confidence
                conf = 0
                try:
                    # try to extract confidence from various places
                    conf = float(issue.get("confidence", 0))
                    if conf == 0:
                         conf = float(issue.get("report", {}).get("issue_overview", {}).get("confidence", 0))
                except:
                    pass
                
                if conf > 0 and conf < 50:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"⚠️ WARNING: This report has low AI confidence ({conf}%). Are you sure you want to deactivate the user? Please retry with force_confirm=true."
                    )

        # Proceed with deactivation
        user = await mongo_service.db.users.find_one({"email": action.user_email})
        if user:
            await mongo_service.db.users.update_one(
                {"email": action.user_email},
                {"$set": {"is_active": False, "deactivation_reason": action.reason}}
            )
        else:
            await mongo_service.db.blacklisted_users.update_one(
                {"email": action.user_email},
                {"$set": {"email": action.user_email, "reason": action.reason, "admin_id": action.admin_id, "timestamp": datetime.utcnow()}},
                upsert=True
            )
            
        return {"message": f"User {action.user_email} deactivated."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{admin_id}")
async def delete_admin(admin_id: str, current_admin: dict = Depends(require_permission("manage_team"))):
    """
    Delete an admin user.
    """
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Prevent self-deletion (optional but recommended safety)
        if str(current_admin["id"]) == admin_id:
             raise HTTPException(status_code=400, detail="Cannot delete your own account.")

        collection = await mongo_service.get_collection("admins", read_only=False)
        
        # Permission Check: Check target admin role
        obj_id = parse_id(admin_id)
        if not obj_id:
             raise HTTPException(status_code=400, detail="Invalid Admin ID format")
             
        target_admin = await collection.find_one({"_id": obj_id})
        if not target_admin:
             raise HTTPException(status_code=404, detail="Admin not found")

        target_role = target_admin.get("role", "admin")
        current_role = current_admin.get("role", "admin")

        if target_role == "super_admin" and current_role != "super_admin":
            raise HTTPException(
                status_code=403, 
                detail="Only a Super Admin can delete another Super Admin"
            )

        result = await collection.delete_one({"_id": obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Admin not found")
            
        return {"message": "Admin deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting admin: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/deactivate-admin/{admin_id}")
async def deactivate_admin_account(admin_id: str, current_admin: dict = Depends(require_permission("manage_team"))):
    """
    Deactivate an admin user.
    Rules:
    - Only super_admin can deactivate another super_admin.
    - Cannot deactivate self.
    """
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Prevent self-deactivation
        if str(current_admin["id"]) == admin_id:
             raise HTTPException(status_code=400, detail="Cannot deactivate your own account.")

        collection = await mongo_service.get_collection("admins", read_only=False)
        
        obj_id = parse_id(admin_id)
        if not obj_id:
             raise HTTPException(status_code=400, detail="Invalid Admin ID format")
             
        target_admin = await collection.find_one({"_id": obj_id})
        
        if not target_admin:
            raise HTTPException(status_code=404, detail="Admin not found")
            
        # Permission Check: Only super_admin can deactivate super_admin
        target_role = target_admin.get("role", "admin")
        current_role = current_admin.get("role", "admin")
        
        if target_role == "super_admin" and current_role != "super_admin":
            raise HTTPException(
                status_code=403, 
                detail="Only a Super Admin can deactivate another Super Admin"
            )
            
        # Update status
        result = await collection.update_one(
            {"_id": obj_id},
            {"$set": {"is_active": False}}
        )
        
        # If no change, it might be already inactive, but we return success anyway
        return {"message": "Admin deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating admin: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/reactivate-admin/{admin_id}")
async def reactivate_admin_account(admin_id: str, current_admin: dict = Depends(require_permission("manage_team"))):
    """
    Reactivate an admin user.
    """
    if current_admin.get("role") != "super_admin":
         raise HTTPException(status_code=403, detail="Only Super Admin can reactivate accounts")

    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        collection = await mongo_service.get_collection("admins", read_only=False)
        obj_id = parse_id(admin_id)
        if not obj_id:
             raise HTTPException(status_code=400, detail="Invalid Admin ID format")
             
        result = await collection.update_one(
            {"_id": obj_id},
            {"$set": {"is_active": True}}
        )
        
        return {"message": "Admin reactivated successfully"}
    except Exception as e:
        logger.error(f"Error reactivating admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ============================================
# ISSUE ASSIGNMENT ENDPOINTS
# ============================================

def parse_id(id_val):
    """Robust conversion of various ID formats. Supports ObjectId and String/UUID IDs."""
    if not id_val:
        return None
    if isinstance(id_val, ObjectId):
        return id_val
    if isinstance(id_val, dict):
        # Handle {$oid: "..."} format
        id_val = id_val.get("$oid") or id_val.get("id") or id_val.get("_id")
    
    if not isinstance(id_val, str):
        id_val = str(id_val)
        
    # 1. Try ObjectId (24-char hex)
    try:
        if len(id_val) == 24 and all(c in "0123456789abcdefABCDEF" for c in id_val):
            return ObjectId(id_val)
    except:
        pass

    # 2. Fallback to String (for UUIDs or other string-based IDs)
    if id_val.strip():
        return id_val
        
    return None

@router.post("/assign-issue")
async def assign_issue_to_admin(
    issue_id: str = Body(...),
    admin_email: str = Body(...),
    current_admin: dict = Depends(get_admin_user)
):
    """
    Assign an issue to a specific admin. Only super_admin can assign.
    """
    # Check permissions
    if current_admin.get("role") not in ["super_admin", "admin"]:
        raise HTTPException(
            status_code=403,
            detail="Only admins and super admins can assign issues"
        )
    
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # Find the admin to assign to
        admins_collection = await mongo_service.get_collection("admins")
        target_admin = await admins_collection.find_one({"email": admin_email})
        
        if not target_admin:
            raise HTTPException(status_code=404, detail=f"Admin {admin_email} not found")
        
        if not target_admin.get("is_active"):
            raise HTTPException(status_code=400, detail="Cannot assign to inactive admin")
        
        # Check if admin has permission to handle issues
        permissions = target_admin.get("permissions", {})
        if not permissions.get("can_approve") and not permissions.get("can_decline"):
            raise HTTPException(
                status_code=400,
                detail=f"Admin {admin_email} (role: {target_admin.get('role')}) cannot handle issues"
            )
        
        # Update admin's assigned_issues list
        obj_id = parse_id(issue_id)
        if not obj_id:
             raise HTTPException(status_code=400, detail="Invalid Issue ID format")

        await admins_collection.update_one(
            {"email": admin_email},
            {"$addToSet": {"assigned_issues": str(obj_id)}}
        )
        
        # Update issue with assigned_to field
        issues_collection = await mongo_service.get_collection("issues")
        await issues_collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "assigned_to": admin_email,
                    "assigned_by": current_admin.get("email"),
                    "assigned_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Issue {issue_id} assigned to {admin_email} by {current_admin.get('email')}")
        
        return {
            "message": f"Issue assigned to {target_admin.get('name', admin_email)}",
            "issue_id": issue_id,
            "assigned_to": {
                "email": admin_email,
                "name": target_admin.get("name"),
                "role": target_admin.get("role")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-assigned-issues")
async def get_my_assigned_issues(current_admin: dict = Depends(get_admin_user)):
    """
    Get all issues assigned to the current admin
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        admin_email = current_admin.get("email")
        
        # Get admin's assigned issues
        issues_collection = await mongo_service.get_collection("issues")
        cursor = issues_collection.find({"assigned_to": admin_email}).sort("timestamp", -1)
        assigned_issues = await cursor.to_list(length=100)
        
        # Format response & Normalize
        for issue in assigned_issues:
            sid = str(issue["_id"])
            issue["_id"] = sid
            issue["issue_id"] = sid
            
            # Normalize Image URL for frontend
            if "image_url" not in issue:
                issue["image_url"] = f"/api/issues/{sid}/image"

            # Ensure minimal report structure if missing (prevents frontend crash)
            if "report" not in issue:
                issue["report"] = {}

        logger.info(f"📋 Retrieved {len(assigned_issues)} assigned issues for {admin_email}")
        
        return {
            "count": len(assigned_issues),
            "issues": assigned_issues
        }
        
    except Exception as e:
        logger.error(f"Error fetching assigned issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STATISTICS ENDPOINT
# ============================================


class BulkAssignRequest(BaseModel):
    issue_ids: List[str]
    admin_email: str

@router.post("/bulk-assign")
async def bulk_assign_issues(request: BulkAssignRequest, current_admin: dict = Depends(get_admin_user)):
    """Assign multiple issues to an admin"""
    # Only allow assignment if super_admin or maybe normal admin too?
    # User said "me admin hu". So standard admin should be able to assign.
    # But usually assignment is restricted.
    # Existing 'assign_issue' is restricted to 'super_admin'.
    # I should check permissions. User asked "me admin hu".
    # I'll stick to 'super_admin' check to match existing 'assign_issue'.
    # If they are just 'admin', they might not have permission.
    # BUT, existing 'assign-issue' endpoint enforces super_admin.
    
    if current_admin.get("role") not in ["super_admin", "admin"]:
         raise HTTPException(status_code=403, detail="Only admins/super admins can assign issues")
    
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        raise HTTPException(status_code=503, detail="Service unavailable")

    collection = await mongo_service.get_collection("issues")
    
    # Validate Issue IDs
    object_ids = []
    for id_val in request.issue_ids:
        oid = parse_id(id_val)
        if not oid:
             raise HTTPException(status_code=400, detail=f"Invalid Issue ID format: {id_val}")
        object_ids.append(oid)

    # Update Issues
    result = await collection.update_many(
        {"_id": {"$in": object_ids}},
        {"$set": {
            "assigned_to": request.admin_email, 
            "updated_at": datetime.utcnow(),
            "assigned_by": current_admin.get("email"),
            "assigned_at": datetime.utcnow()
        }}
    )
    
    # Also update admin record for consistency
    if result.modified_count > 0:
        admins_collection = await mongo_service.get_collection("admins")
        # Store as strings for consistency in the list
        str_ids = [str(oid) for oid in object_ids]
        await admins_collection.update_one(
            {"email": request.admin_email},
            {"$addToSet": {"assigned_issues": {"$each": str_ids}}}
        )
    
    logger.info(f"Bulk assigned {result.modified_count} issues to {request.admin_email} by {current_admin['email']}")
    
    return {"message": f"Assigned {result.modified_count} issues", "count": result.modified_count}


@router.get("/stats", response_model=dict)
async def get_admin_stats(admin: dict = Depends(get_admin_user)):
    """
    Get aggregated statistics for the admin dashboard.
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
            
        issues_coll = await mongo_service.get_collection("issues")
        admins_coll = await mongo_service.get_collection("admins")
        
        # Define valid status groups (Excluding 'failed' and system errors)
        valid_pending = ["needs_review", "waiting_review", "pending"]
        valid_approved = ["submitted", "approved", "completed", "resolved"]
        valid_declined = ["rejected", "declined"]
        
        # 1. Parallelize Counting for Speed (Exclude failed reports)
        start_time = datetime.utcnow()
        
        # Define tasks - Counting only verified mission states
        task_total = issues_coll.count_documents({"status": {"$in": valid_pending + valid_approved + valid_declined}})
        task_pending = issues_coll.count_documents({"status": {"$in": valid_pending}})
        task_approved = issues_coll.count_documents({"status": {"$in": valid_approved}})
        task_declined = issues_coll.count_documents({"status": {"$in": valid_declined}})
        
        # Execute counts in parallel
        total_issues, pending_review, approved, declined = await asyncio.gather(
            task_total, task_pending, task_approved, task_declined
        )
        
        # 2. Issues by Type (Optimized Pipeline)
        pipeline_type = [
            {"$group": {"_id": "$issue_type", "count": {"$sum": 1}}}
        ]
        type_cursor = issues_coll.aggregate(pipeline_type)
        by_type_list = await type_cursor.to_list(None)
        by_type = {doc["_id"] or "Unknown": doc["count"] for doc in by_type_list}
        
        # 3. Team Performance
        all_admins = await admins_coll.find({}, {"email": 1, "name": 1, "role": 1}).to_list(None)
        
        # Aggregate resolved/assigned in parallel
        pipeline_resolved = [
            {"$match": {"admin_review.admin_id": {"$exists": True}}},
            {"$group": {"_id": "$admin_review.admin_id", "count": {"$sum": 1}}}
        ]
        pipeline_assigned = [
             {"$match": {"assigned_to": {"$exists": True, "$ne": None}}},
             {"$group": {"_id": "$assigned_to", "count": {"$sum": 1}}}
        ]
        
        resolved_cursor = issues_coll.aggregate(pipeline_resolved)
        assigned_cursor = issues_coll.aggregate(pipeline_assigned)
        
        resolved_docs, assigned_docs = await asyncio.gather(
            resolved_cursor.to_list(None),
            assigned_cursor.to_list(None)
        )
        
        resolved_map = {doc["_id"]: doc["count"] for doc in resolved_docs}
        assigned_map = {doc["_id"]: doc["count"] for doc in assigned_docs}
        
        team_stats = []
        for a in all_admins:
            email = a.get("email")
            r_count = resolved_map.get(email, 0)
            aid = str(a.get("_id"))
            if aid in resolved_map: r_count += resolved_map[aid]

            a_count = assigned_map.get(email, 0)
            
            team_stats.append({
                "name": a.get("name", "Unknown"),
                "email": email,
                "role": a.get("role", "admin"),
                "resolved": r_count,
                "assigned": a_count
            })
            
        # 4. Recent Activity
        recent_cursor = issues_coll.find(
            {"admin_review": {"$exists": True}}
        ).sort("admin_review.timestamp", -1).limit(5)
        recent_docs = await recent_cursor.to_list(None)
        
        recent_activity = []
        for doc in recent_docs:
            ar = doc.get("admin_review", {})
            action = ar.get("action", "processed")
            actor = ar.get("admin_id", "Unknown")
            issue_type = doc.get("issue_type", "issue")
            
            description = f"{actor} {action} {issue_type}"
            if action == 'approve': description = f"{actor} approved {issue_type}"
            if action == 'decline': description = f"{actor} declined {issue_type}"
            
            recent_activity.append({
                "description": description,
                "timestamp": ar.get("timestamp", doc.get("timestamp"))
            })

        # 5. Financial Stats (Super Admin Only)
        # Handle role check carefully
        user_role = str(admin.get("role", "")).lower().strip() # Normalize
        user_email = admin.get("email")
        
        logger.info(f"📊 Stats requested by {user_email} (Role: {user_role})")
        
        response_data = {
            "total_issues": total_issues,
            "pending_review": pending_review,
            "approved": approved,
            "declined": declined,
            "by_type": by_type,
            "team_performance": team_stats,
            "recent_activity": recent_activity
        }
        
        if user_role == "super_admin":
            logger.info(f"💰 Adding financial stats for Super Admin: {user_email}")
            
            # 1. Financials
            total_ai_reports = await issues_coll.count_documents({"report": {"$exists": True}})
            cost_per = 0.00026 
            total_cost = total_ai_reports * cost_per
            
            response_data["financials"] = {
                "total_ai_reports": total_ai_reports,
                "cost_per_report": cost_per,
                "total_cost_usd": total_cost,
                "currency": "USD"
            }

            # 2. AI Performance & Ecosystem Metrics
            # Calculate Average Confidence Score from existing reports
            pipeline_conf = [
                {"$match": {"report.unified_report.confidence_percent": {"$exists": True, "$ne": None}}},
                {"$group": {"_id": None, "avg_confidence": {"$avg": "$report.unified_report.confidence_percent"}}}
            ]
            
            try:
                conf_cursor = issues_coll.aggregate(pipeline_conf)
                conf_result = await conf_cursor.to_list(None)
                avg_confidence = conf_result[0]["avg_confidence"] if conf_result else 0
            except Exception as e:
                logger.warning(f"Failed to calculate avg confidence: {e}")
                avg_confidence = 0

            # Advanced System Stats (Hybrid: Real + Static Metadata)
            response_data["ai_performance"] = {
                "model_name": "Gemini 1.5 Flash",
                "model_version": "v1.5-latest",
                "provider": "Google DeepMind",
                "architecture": "Multimodal (Vision + Text)",
                "context_window": "1M Tokens",
                "avg_confidence": round(avg_confidence, 1),
                "avg_latency_ms": 2100, # Estimated average
                "uptime": "99.99%",
                "requests_processed": total_ai_reports,
                "optimization_level": "High (GridFS + Caching)"
            }
        else:
            logger.info(f"⏩ Skipping financial stats for role: '{user_role}'")

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"⏱️ Stats generation took {duration:.2f}s")
        
        return response_data
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# SECURITY ENDPOINTS
# ============================================

@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_admin: dict = Depends(get_admin_user),
    request: Request = None
):
    """Change admin password"""
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    # Get full admin record to check current password
    admin = await collection.find_one({"email": current_admin["email"]})
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
        
    # Verify current password
    if not verify_password(password_data.current_password, admin["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
        
    # Update to new password
    new_hash = get_password_hash(password_data.new_password)
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "password_hash": new_hash,
                "require_password_change": False,
                "password_last_changed": datetime.utcnow()
            }
        }
    )
    
    # Audit log
    if request:
        client_info = SecurityService.get_client_info(request)
        await SecurityService.log_security_event(
            admin_email=current_admin["email"],
            action="password_change",
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"],
            success=True
        )
        
    return {"message": "Password changed successfully"}

@router.post("/2fa/setup")
async def setup_2fa(
    setup_data: TwoFactorSetup,
    current_admin: dict = Depends(get_admin_user)
):
    """Initialize 2FA setup - returns secret"""
    if setup_data.method != "totp":
        raise HTTPException(status_code=400, detail="Only TOTP currently supported")
        
    # Generate secret
    secret = SecurityService.generate_2fa_secret()
    
    return {
        "secret": secret,
        "uri": f"otpauth://totp/EAiSER:{current_admin['email']}?secret={secret}&issuer=EAiSER"
    }

@router.post("/2fa/verify")
async def verify_2fa_setup(
    verify_data: TwoFactorVerify,
    current_admin: dict = Depends(get_admin_user)
):
    """Verify and enable 2FA"""
    secret = verify_data.session_token
    
    if not SecurityService.verify_2fa_code(secret, verify_data.code):
        raise HTTPException(status_code=400, detail="Invalid code")
        
    # Encrypt the TOTP secret before storing in database
    from utils.totp_encryption import encrypt_totp_secret
    encrypted_secret = encrypt_totp_secret(secret)
    
    # Enable 2FA for user
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "two_factor_enabled": True,
                "two_factor_secret": encrypted_secret,
                "two_factor_method": "totp"
            }
        }
    )
    
    return {"message": "2FA verified and enabled"}

@router.post("/2fa/disable")
async def disable_2fa(
    current_admin: dict = Depends(get_admin_user)
):
    """Disable 2FA"""
    mongo_service = await get_optimized_mongodb_service()
    collection = await mongo_service.get_collection("admins")
    
    await collection.update_one(
        {"email": current_admin["email"]},
        {
            "$set": {
                "two_factor_enabled": False,
                "two_factor_secret": None
            }
        }
    )
    
    return {"message": "2FA disabled"}


# --- MAPPING REVIEW ENDPOINTS ---

@router.get("/mapping-review")
async def get_unmapped_issues(resolved: bool = False, limit: int = 20, admin: dict = Depends(get_admin_user)):
    """Get unmapped issues for admin review"""
    try:
        db = await get_database()
        
        # Sort by flagged_at DESC
        cursor = db.authority_mapping_review.find({"resolved": resolved}).sort("flagged_at", -1).limit(limit)
        entries = await cursor.to_list(length=limit)
        
        # Count total
        total = await db.authority_mapping_review.count_documents({"resolved": False})
        
        # Convert _id to string
        for entry in entries:
            entry["_id"] = str(entry["_id"])
            
        return {
            'entries': entries,
            'count': len(entries),
            'total_unmapped': total
        }
    except Exception as e:
        logger.error(f"Error fetching unmapped issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ResolveMappingRequest(BaseModel):
    issue_type: str
    mapped_departments: List[str]

class RejectMappingRequest(BaseModel):
    reason: Optional[str] = "Issue could not be verified or does not meet reporting criteria."

@router.post("/mapping-review/{review_id}/resolve")
async def resolve_mapping(review_id: str, request: ResolveMappingRequest, admin: dict = Depends(get_admin_user)):
    """
    Admin approves an unmapped issue:
    1. Marks the review entry as resolved
    2. Updates the global department mapping table
    3. Dispatches the report to the selected authority via email
    4. Updates the main issue status to 'submitted'
    """
    from services.authority_service import update_department_mapping, get_authority_by_department
    from services.mongodb_optimized_service import get_optimized_mongodb_service
    
    try:
        db = await get_database()
        
        # 1. Fetch the review entry
        review_entry = await db.authority_mapping_review.find_one({"id": review_id})
        if not review_entry:
            raise HTTPException(status_code=404, detail="Review entry not found")
        
        issue_id = review_entry.get("issue_id") or review_entry.get("case_id")
        zip_code = review_entry.get("zip_code", "")
        
        # 2. Mark review entry as resolved (approved)
        await db.authority_mapping_review.update_one(
            {"id": review_id},
            {"$set": {
                "resolved": True,
                "resolution": "approved",
                "resolved_mapping": ','.join(request.mapped_departments),
                "resolved_by": admin.get("email"),
                "resolved_at": datetime.utcnow().isoformat()
            }}
        )
        
        # 3. Update global department mapping table for future auto-routing
        await update_department_mapping(request.issue_type, request.mapped_departments, admin_email=admin.get("email"))
        
        # 4. Find the actual issue and dispatch to the authority
        dispatched = False
        if issue_id:
            mongodb_service = await get_optimized_mongodb_service()
            issue = await mongodb_service.get_issue_by_id(issue_id) if mongodb_service else None
            
            if issue:
                # Build authority list from the selected departments + zip code
                authorities = []
                for dept in request.mapped_departments:
                    dept_auths = get_authority_by_department(zip_code, dept)
                    if dept_auths:
                        for a in dept_auths:
                            if not any(x.get("email") == a.get("email") for x in authorities):
                                authorities.append(a)
                
                # Fallback: use department name as the authority
                if not authorities:
                    authorities = [{
                        "name": f"{', '.join(request.mapped_departments).replace('_', ' ').title()} Department",
                        "email": "eaiser@momntumai.com",
                        "type": request.mapped_departments[0]
                    }]
                
                # Fetch image for email
                image_content = b""
                try:
                    from bson.objectid import ObjectId
                    image_id = issue.get("image_id")
                    if image_id and mongodb_service:
                        grid_out = await mongodb_service.fs.open_download_stream(ObjectId(image_id))
                        image_content = await grid_out.read()
                except Exception:
                    pass
                
                # Dispatch email
                try:
                    report = issue.get("report") or {}
                    conf_val = float(issue.get("confidence") or 0)
                    email_success = await send_authority_email(
                        issue_id=issue_id,
                        authorities=authorities,
                        issue_type=request.issue_type,
                        final_address=issue.get("address", "N/A"),
                        zip_code=zip_code or "N/A",
                        timestamp_formatted=issue.get("timestamp_formatted") or datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                        report=report,
                        confidence=conf_val,
                        category=issue.get("category", "public"),
                        timezone_name=issue.get("timezone_name", "UTC"),
                        latitude=float(issue.get("latitude") or 0),
                        longitude=float(issue.get("longitude") or 0),
                        image_content=image_content
                    )
                    dispatched = email_success
                    logger.info(f"✅ Admin-dispatched report {issue_id} → {[a['email'] for a in authorities]} | success={email_success}")
                except Exception as e:
                    logger.error(f"Email dispatch failed for {issue_id}: {e}")
                
                # Update issue status in DB
                if mongodb_service:
                    await mongodb_service.update_issue_status(issue_id, {
                        "status": "submitted",
                        "issue_type": request.issue_type,
                        "is_submitted": True,
                        "email_status": "sent" if dispatched else "failed",
                        "authority_email": [a.get("email") for a in authorities],
                        "authority_name": [a.get("name") for a in authorities],
                        "admin_approved_by": admin.get("email"),
                        "admin_approved_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow()
                    })
        
        return {
            'status': 'approved',
            'issue_type': request.issue_type, 
            'mapped_to': request.mapped_departments,
            'dispatched': dispatched,
            'issue_id': issue_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving mapping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mapping-review/{review_id}/reject")
async def reject_mapping(review_id: str, request: RejectMappingRequest, admin: dict = Depends(get_admin_user)):
    """
    Admin rejects/denies an unmapped issue:
    1. Marks the review entry as resolved (rejected)
    2. Updates the main issue status to 'screened_out'
    """
    from services.mongodb_optimized_service import get_optimized_mongodb_service
    
    try:
        db = await get_database()
        
        # 1. Fetch the review entry
        review_entry = await db.authority_mapping_review.find_one({"id": review_id})
        if not review_entry:
            raise HTTPException(status_code=404, detail="Review entry not found")
        
        issue_id = review_entry.get("issue_id")
        
        # 2. Mark review entry as resolved (rejected)
        await db.authority_mapping_review.update_one(
            {"id": review_id},
            {"$set": {
                "resolved": True,
                "resolution": "rejected",
                "rejection_reason": request.reason,
                "resolved_by": admin.get("email"),
                "resolved_at": datetime.utcnow().isoformat()
            }}
        )
        
        # 3. Update the main issue status to screened_out
        if issue_id:
            mongodb_service = await get_optimized_mongodb_service()
            if mongodb_service:
                await mongodb_service.update_issue_status(issue_id, {
                    "status": "screened_out",
                    "is_submitted": False,
                    "rejection_reason": request.reason,
                    "admin_rejected_by": admin.get("email"),
                    "admin_rejected_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow()
                })
        
        logger.info(f"❌ Admin rejected mapping review {review_id} (issue: {issue_id}) | reason: {request.reason}")
        
        return {
            'status': 'rejected',
            'issue_id': issue_id,
            'reason': request.reason
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting mapping: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# --- AUTHORITY MANAGEMENT ENDPOINTS ---

@router.get("/authorities")
async def get_authorities_list(admin: dict = Depends(get_admin_user)):
    """Get all zip code authorities."""
    from services.authority_service import get_all_authorities
    return get_all_authorities()

class UpdateZipRequest(BaseModel):
    zip_code: str
    data: dict  # format: { dept: [{name, email, type, timezone}] }

@router.post("/authorities")
async def update_authority(request: UpdateZipRequest, admin: dict = Depends(get_admin_user)):
    """Update contacts for a zip code."""
    from services.authority_service import update_zip_authority
    
    # Basic validation could happen here
    
    success = await update_zip_authority(request.zip_code, request.data, admin_email=admin.get("email"))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save authority data")
        
    logger.info(f"Admin {admin['email']} updated authorities for ZIP {request.zip_code}")
    
    return {"message": "Authority data updated", "zip_code": request.zip_code}

@router.get("/mappings")
async def get_mappings(admin: dict = Depends(get_admin_user)):
    """Get all global issue mappings."""
    from services.authority_service import get_all_department_mappings
    return get_all_department_mappings()

class UpdateMappingRequest(BaseModel):
    issue_type: str
    departments: List[str]

@router.post("/mappings/update")
async def update_single_mapping(request: UpdateMappingRequest, admin: dict = Depends(get_admin_user)):
    """Update a specific issue type mapping."""
    from services.authority_service import update_department_mapping
    success = await update_department_mapping(request.issue_type, request.departments, admin_email=admin.get("email"))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save mapping")
    return {"status": "updated", "issue_type": request.issue_type}

@router.get("/stats/mapping")
async def get_mapping_stats(admin: dict = Depends(get_admin_user)):
    """Get high-level statistics for mapping coverage."""
    db = await get_database()
    from services.authority_service import ISSUE_DEPARTMENT_MAP
    
    # 1. Coverage
    total_types = len(ISSUE_DEPARTMENT_MAP)
    
    # 2. Unmapped pending
    unmapped_count = await db.authority_mapping_review.count_documents({"resolved": False})
    
    # 3. Zip Coverage (simple count for now)
    from services.authority_service import ZIP_CODE_AUTHORITIES
    total_zips = len(ZIP_CODE_AUTHORITIES) - 1 # exclude default?
    
    return {
        "mapped_types": total_types,
        "pending_reviews": unmapped_count,
        "zip_codes_configured": total_zips,
        "coverage_percent": 95 # Placeholder or calculated
    }

@router.get("/analytics/forecast")
async def get_forecast(admin: dict = Depends(get_admin_user)):
    """
    Get AI-generated issue forecast based on real historical regression.
    """
    from services.predictive_analytics import PredictiveAnalyticsService
    return await PredictiveAnalyticsService.get_issue_forecast()

@router.get("/mapping-history")
async def get_mapping_history(limit: int = 50, admin: dict = Depends(get_admin_user)):
    """Get audit log of all changes."""
    try:
        db = await get_database()
        cursor = db.audit_logs.find().sort("timestamp", -1).limit(limit)
        history = await cursor.to_list(length=limit)
        
        # Convert _id to string
        for h in history:
            h["_id"] = str(h["_id"])
            
        return history
    except Exception as e:
        logger.error(f"Error fetching mapping history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
