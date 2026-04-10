from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import Optional
import os
import asyncio
import logging
from services.ai_service import classify_issue

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ai/analyze-image")
async def analyze_image(request: Request):
    """
    Bridge to unified AI V3 Brain for consistent behavior between mobile and web.
    Accepts multipart form data with an 'image' file field.
    """
    try:
        from starlette.datastructures import UploadFile as StarletteUploadFile
        
        # Manually parse multipart to be resilient to missing optional fields
        form = await request.form()
        
        # Try 'image' first, then 'file' as fallback
        target_file = form.get("image") or form.get("file")
        description = str(form.get("description", "") or "")
        
        if not target_file:
            logger.warning(f"❌ AI Analysis: No image field found. Available fields: {list(form.keys())}")
            return {
                "status": "error",
                "message": f"No image provided. Received fields: {list(form.keys())}",
                "issue_type": "unknown",
                "severity": "medium",
                "confidence": 0,
                "issue_detected": False
            }

        # Starlette form() returns UploadFile for file fields, str for text fields
        if isinstance(target_file, StarletteUploadFile):
            content = await target_file.read()
            filename = target_file.filename or 'unknown'
        elif isinstance(target_file, bytes):
            content = target_file
            filename = 'raw_bytes'
        else:
            logger.error(f"❌ AI Analysis: 'image' field is type {type(target_file).__name__}, not a file. Value preview: {str(target_file)[:100]}")
            return {
                "status": "error",
                "message": "The 'image' field was not sent as a file upload.",
                "issue_type": "unknown",
                "severity": "medium",
                "confidence": 0,
                "issue_detected": False
            }

        logger.info(f"🧠 AI Preview Analysis (V3 Brain) for {filename}: {len(content)} bytes")

        # Call the unified V3 Brain (same as Website)
        issue_type, severity, confidence, category, priority, issue_detected = await classify_issue(content, str(description))

        # Format for Mobile/Web Preview screen
        return {
            "issue_type": issue_type,
            "severity": severity,
            "confidence": confidence,
            "category": category,
            "priority": priority,
            "issue_detected": issue_detected,
            "description": f"EAiSER AI identified this as a potential {issue_type} with high precision.",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"AI Preview Analysis failed: {str(e)}", exc_info=True)
        # Fallback to manual route
        return {
            "issue_type": "unknown",
            "severity": "medium",
            "confidence": 0,
            "description": "AI analysis is taking longer than expected. You can still report manually.",
            "status": "fallback"
        }

@router.post("/analyze-image")
async def analyze_image_alias(request: Request):
    return await analyze_image(request)
