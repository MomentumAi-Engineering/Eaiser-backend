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
        issue_type, severity, confidence, category, priority, issue_detected, *_simi_extra = await classify_issue(content, str(description))
        simi_data = _simi_extra[0] if _simi_extra else None

        # Use the REAL AI scene narrative — only when it actually reads like prose.
        # We REJECT comma-joined label lists (e.g. "fallen_tree, road_damage") which
        # have spaces and meet length but aren't real descriptions.
        def _is_sentence_like(s: str) -> bool:
            t = (s or "").strip()
            if len(t) < 25:
                return False
            if "_" in t:               # snake_case label artifacts
                return False
            if len(t.split()) < 5:     # need at least ~5 words for a real sentence
                return False
            # Must contain at least one common connector word (any prose sentence will)
            connectors = (" is ", " was ", " has ", " have ", " the ", " a ", " an ",
                          " on ", " at ", " with ", " near ", " from ", " and ",
                          " or ", " of ", " in ", " across ", " over ", " under ")
            lower = " " + t.lower() + " "
            return any(c in lower for c in connectors)

        scene_summary = ""
        if simi_data:
            scene = (simi_data.get("scene_description") or "").strip()
            if _is_sentence_like(scene):
                scene_summary = scene
            # We intentionally do NOT fall back to issue_summary — it's a label list,
            # never a sentence; the UI placeholder is a better experience than that.
        # Empty is fine — UI will show the placeholder so the user can add their own context.

        # Format for Mobile/Web Preview screen
        response = {
            "issue_type": issue_type,
            "severity": severity,
            "confidence": confidence,
            "category": category,
            "priority": priority,
            "issue_detected": issue_detected,
            "description": scene_summary,
            "status": "success"
        }

        # 🆕 SIMI Level 3: Include all detected issues for mobile preview.
        # scene_description = AI's actual scene narrative (sentences).
        # issue_summary    = comma-joined list of detected issue labels.
        if simi_data:
            response["detected_issues"] = simi_data.get("ordered_issue_list", [])
            response["total_detected_issues"] = simi_data.get("total_issues", 1)
            response["scene_description"] = simi_data.get("scene_description", "")
            response["issue_summary"] = simi_data.get("issue_summary", "")
            response["emergency_911"] = simi_data.get("emergency_911", False)
            response["emergency_advisory"] = simi_data.get("emergency_advisory")
            response["scenario"] = simi_data.get("scenario", "A")
        
        return response
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
