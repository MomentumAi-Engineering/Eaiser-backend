from fastapi import APIRouter, HTTPException, Depends, Body, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging
from services.mongodb_optimized_service import get_optimized_mongodb_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analytics",
    tags=["Analytics"]
)

class CookieConsentLog(BaseModel):
    action: str  # 'accept_all', 'reject_optional', 'save_preferences'
    preferences: dict
    user_agent: Optional[str] = None
    ip_hash: Optional[str] = None # For anonymized tracking

@router.post("/cookie-consent")
async def log_cookie_consent(log: CookieConsentLog, request: Request):
    """Logs cookie consent choice to the database."""
    try:
        mongo = await get_optimized_mongodb_service()
        if not mongo:
            raise HTTPException(status_code=503, detail="Database unavailable")
            
        collection = await mongo.get_collection("cookie_analytics", read_only=False)
        
        # Anonymize IP if possible (simple hash)
        import hashlib
        client_ip = request.client.host
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:12]
        
        log_entry = {
            "action": log.action,
            "preferences": log.preferences,
            "timestamp": datetime.utcnow(),
            "ip_hash": ip_hash,
            "user_agent": request.headers.get("user-agent")
        }
        
        await collection.insert_one(log_entry)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to log cookie consent: {e}")
        # Don't fail the request for the user, just return success anyway or log it
        return {"status": "error", "message": str(e)}

@router.get("/cookie-stats")
async def get_cookie_stats():
    """Returns strictly advanced aggregated stats for cookie consents."""
    try:
        mongo = await get_optimized_mongodb_service()
        if not mongo:
            raise HTTPException(status_code=503, detail="Database unavailable")
            
        collection = await mongo.get_collection("cookie_analytics", read_only=True)
        
        now = datetime.utcnow()
        seven_days_ago = now - timedelta(days=7)
        
        pipeline = [
            {
                "$facet": {
                    "funnel": [
                        {"$group": {
                            "_id": None,
                            "impressions": {"$sum": {"$cond": [{"$eq": ["$action", "impression"]}, 1, 0]}},
                            "interactions": {"$sum": {"$cond": [{"$ne": ["$action", "impression"]}, 1, 0]}},
                            "conversions": {"$sum": {"$cond": [{"$eq": ["$action", "accept_all"]}, 1, 0]}}
                        }}
                    ],
                    "actions": [
                        {"$match": {"action": {"$ne": "impression"}}},
                        {"$group": {"_id": "$action", "count": {"$sum": 1}}}
                    ],
                    "preferences_breakdown": [
                        {"$match": {"action": {"$ne": "impression"}}},
                        {"$group": {
                            "_id": None,
                            "optional_accepted": {"$sum": {"$cond": [{"$eq": ["$preferences.optional", True]}, 1, 0]}},
                            "optional_rejected": {"$sum": {"$cond": [{"$eq": ["$preferences.optional", False]}, 1, 0]}}
                        }}
                    ],
                    "daily_trend": [
                        {"$match": {"timestamp": {"$gte": seven_days_ago}, "action": {"$ne": "impression"}}},
                        {"$group": {
                            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                            "count": {"$sum": 1},
                            "accepted": {"$sum": {"$cond": [{"$eq": ["$preferences.optional", True]}, 1, 0]}}
                        }},
                        {"$sort": {"_id": 1}}
                    ],
                    "devices": [
                        {"$match": {"action": {"$ne": "impression"}}},
                        {"$group": {
                            "_id": {
                                "$cond": [
                                    {"$regexMatch": {"input": "$user_agent", "regex": "Mobile|Android|iPhone", "options": "i"}},
                                    "Mobile",
                                    "Desktop"
                                ]
                            },
                            "count": {"$sum": 1}
                        }}
                    ],
                    "browsers": [
                        {"$match": {"action": {"$ne": "impression"}}},
                        {"$group": {
                            "_id": {
                                "$cond": [
                                    {"$regexMatch": {"input": "$user_agent", "regex": "Edg", "options": "i"}}, "Edge",
                                    {"$cond": [
                                        {"$regexMatch": {"input": "$user_agent", "regex": "Chrome", "options": "i"}}, "Chrome",
                                        {"$cond": [
                                            {"$regexMatch": {"input": "$user_agent", "regex": "Safari", "options": "i"}}, "Safari",
                                            {"$cond": [
                                                {"$regexMatch": {"input": "$user_agent", "regex": "Firefox", "options": "i"}}, "Firefox",
                                                "Other"
                                            ]}
                                        ]}
                                    ]}
                                ]
                            },
                            "count": {"$sum": 1}
                        }}
                    ],
                    "os_distribution": [
                        {"$match": {"action": {"$ne": "impression"}}},
                        {"$group": {
                            "_id": {
                                "$cond": [
                                    {"$regexMatch": {"input": "$user_agent", "regex": "Windows", "options": "i"}}, "Windows",
                                    {"$cond": [
                                        {"$regexMatch": {"input": "$user_agent", "regex": "Macintosh|Mac OS", "options": "i"}}, "MacOS",
                                        {"$cond": [
                                            {"$regexMatch": {"input": "$user_agent", "regex": "Android", "options": "i"}}, "Android",
                                            {"$cond": [
                                                {"$regexMatch": {"input": "$user_agent", "regex": "iPhone|iPad", "options": "i"}}, "iOS",
                                                "Linux/Other"
                                            ]}
                                        ]}
                                    ]}
                                ]
                            },
                            "count": {"$sum": 1}
                        }}
                    ]
                }
            }
        ]
        
        results = await collection.aggregate(pipeline).to_list(1)
        res = results[0] if results else {}
        
        funnel_list = res.get("funnel", [{}])
        funnel = funnel_list[0] if funnel_list else {"impressions": 0, "interactions": 0, "conversions": 0}
        
        actions = {item["_id"]: item["count"] for item in res.get("actions", [])}
        prefs_list = res.get("preferences_breakdown", [{}])
        prefs = prefs_list[0] if prefs_list else {}
        
        return {
            "funnel": funnel,
            "total_interactions": funnel.get("interactions", 0),
            "actions": actions,
            "preferences": {
                "essential": funnel.get("interactions", 0),
                "optional_accepted": prefs.get("optional_accepted", 0),
                "optional_rejected": prefs.get("optional_rejected", 0)
            },
            "daily_trend": res.get("daily_trend", []),
            "device_breakdown": {item["_id"]: item["count"] for item in res.get("devices", [])},
            "browser_breakdown": {item["_id"]: item["count"] for item in res.get("browsers", [])},
            "os_breakdown": {item["_id"]: item["count"] for item in res.get("os_distribution", [])}
        }
    except Exception as e:
        logger.error(f"Failed to get cookie stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cookie-logs")
async def get_recent_cookie_logs(limit: int = 50):
    """Returns the most recent cookie consent logs."""
    try:
        mongo = await get_optimized_mongodb_service()
        if not mongo:
            raise HTTPException(status_code=503, detail="Database unavailable")
            
        collection = await mongo.get_collection("cookie_analytics", read_only=True)
        
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                log["timestamp"] = log["timestamp"].isoformat()
        
        return logs
    except Exception as e:
        logger.error(f"Failed to get cookie logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
