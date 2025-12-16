from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import io

# Import the Fake Detector Service
# Assuming models.fake_detector.fake_detector exposes a get_fake_detector or similar
from models.fake_detector.fake_detector import get_fake_detector

# Services for Report Generation
from services.ai_service_optimized import generate_report_optimized

logger = logging.getLogger(__name__)

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import io
import uuid

# Import the Fake Detector Service
from models.fake_detector.fake_detector import get_fake_detector

# Services for Report Generation
from services.ai_service import classify_issue, generate_report

logger = logging.getLogger(__name__)

router = APIRouter()

class AIAnalysisResponse(BaseModel):
    is_real: bool
    confidence: float
    category: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    priority: Optional[str] = None
    details: Dict[str, Any] = {}

class GenerateReportRequest(BaseModel):
    issue_type: str
    severity: str
    description: Optional[str] = ""
    address: Optional[str] = ""
    zip_code: Optional[str] = None
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    category: str
    priority: str
    confidence: float
    # We might need to handle image upload separately or pass image_url if supported
    # validating Python service expects bytes usually.
    # For now, we will handle Report Generation via Form Data or separate endpoint that accepts file.

@router.post("/analyze-image", response_model=AIAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    description: Optional[str] = Form("")
):
    """
    Dedicated endpoint for Node.js Backend to check if an image is Real or Fake,
    and if real, classify it.
    """
    try:
        # Read image bytes
        image_content = await image.read()
        
        # 1. Fake Detection
        fake_detector = get_fake_detector()
        result = await fake_detector.detect_fake(image_content)
        
        is_fake = result.get("is_fake", False)
        # Confidence of being FAKE
        fake_confidence = result.get("confidence", 0.0)

        # Logic for Real/Fake
        # If Fake Confidence > Threshold (e.g. 80), it's fake.
        # But Node.js expects "confidence" of the *result* (Real or Fake).
        
        is_real = not is_fake
        
        response_data = {
            "is_real": is_real,
            "confidence": fake_confidence if is_fake else 0, # Placeholder, will update below
            "details": result
        }

        if is_real:
            # 2. Classification (only if real)
            # classify_issue returns: issue_type, severity, confidence, category, priority
            issue_type, severity, conf, category, priority = await classify_issue(image_content, description or "")
            
            response_data["category"] = category
            response_data["issue_type"] = issue_type
            response_data["severity"] = severity
            response_data["priority"] = priority
            # Use classification confidence for Real result
            response_data["confidence"] = conf
        else:
             # It is fake
             response_data["confidence"] = fake_confidence

        logger.info(f"AI Check Result: Real={is_real}, Category={response_data.get('category')}")

        return response_data

    except Exception as e:
        logger.error(f"Error in analyze-image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-report")
async def generate_report_endpoint(
    image: UploadFile = File(...),
    issue_type: str = Form(...),
    severity: str = Form(...),
    description: str = Form(""),
    address: str = Form(""),
    zip_code: str = Form(None),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    category: str = Form(...),
    priority: str = Form(...),
    confidence: float = Form(...)
):
    """
    Endpoint to generate a full report.
    """
    try:
        image_content = await image.read()
        issue_id = str(uuid.uuid4())
        
        report = await generate_report(
            image_content=image_content,
            description=description,
            issue_type=issue_type,
            severity=severity,
            address=address,
            zip_code=zip_code,
            latitude=latitude,
            longitude=longitude,
            issue_id=issue_id,
            confidence=confidence,
            category=category,
            priority=priority
        )
        
        return report

    except Exception as e:
        logger.error(f"Error in generate-report endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

