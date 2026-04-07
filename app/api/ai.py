from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import asyncio
import logging
from services.ai_service import classify_issue

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ai/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Bridge to unified AI V3 Brain for consistent behavior between mobile and web.
    """
    try:
        content = await image.read()
        logger.info(f"🧠 AI Preview Analysis (V3 Brain) for image size: {len(content)} bytes")

        # Call the unified V3 Brain (same as Website)
        issue_type, severity, confidence, category, priority, issue_detected = await classify_issue(content, "")

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
async def analyze_image_alias(image: UploadFile = File(...)):
    return await analyze_image(image=image)
