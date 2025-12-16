from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import os
import shutil
from pathlib import Path
from datetime import datetime
import uuid
from services.mongodb_service import get_db
from models.report_model import ReportModel, Location, FlagFlowDetails, RealFlowDetails
# Import AI services
from models.fake_detector.fake_detector import get_fake_detector
from services.ai_service import classify_issue, generate_report
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/reports/analyze")
async def analyze_report(
    image: UploadFile = File(...),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    streetAddress: Optional[str] = Form(None),
    zipCode: Optional[str] = Form(None),
    description: Optional[str] = Form("")
):
    try:
        # 1. Save File
        filename = f"{uuid.uuid4()}_{image.filename}"
        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # URL for frontend to display (assuming mounted at /uploads)
        image_url = f"http://localhost:8000/uploads/{filename}"
        
        # 2. AI Check (Real vs Fake)
        fake_detector = get_fake_detector()
        with open(file_path, "rb") as f:
            image_content = f.read()
            
        fake_result = await fake_detector.detect_fake(image_content)
        is_fake = fake_result.get("is_fake", False)
        fake_confidence = fake_result.get("confidence", 0.0) # 0-100 usually
        
        # Normalize confidence logic
        # Node logic: if fake, confidence is high for fake.
        # We need "Real Confidence".
        # If is_fake is True, Real Confidence is Low.
        # If is_fake is False, Real Confidence is High? We need classification confidence.
        
        is_real = not is_fake
        final_confidence = 0.0
        
        category = "public"
        issue_type = "unknown"
        severity = "medium"
        priority = "medium"
        
        if is_real:
            # Classify
            c_issue, c_severity, c_conf, c_category, c_priority = await classify_issue(image_content, description)
            category = c_category
            issue_type = c_issue
            severity = c_severity
            priority = c_priority
            final_confidence = c_conf / 100.0 if c_conf > 1 else c_conf # Normalize to 0-1
        else:
            # It is fake
            # If fake_confidence is 90%, then real confidence is 10% (0.1)
            # Or just set it low.
            final_confidence = 0.05
        
        # 3. Create Report Object
        location = Location(
            latitude=latitude,
            longitude=longitude,
            streetAddress=streetAddress or "",
            zipCode=zipCode or ""
        )
        
        report = ReportModel(
            imageUrl=image_url,
            originalName=image.filename,
            location=location,
            checkResult={
                "isReal": is_real,
                "confidence": final_confidence,
                "fake_score": fake_confidence
            }
        )
        
        # 4. Decision Logic (Strict > 0.70)
        CONFIDENCE_THRESHOLD = 0.70
        
        if is_real and final_confidence > CONFIDENCE_THRESHOLD:
            # High Confidence Real -> Draft
            context_data = {
                "category": category,
                "issue_type": issue_type,
                "severity": severity,
                "priority": priority,
                "confidence": final_confidence,
                "description": description,
                "address": streetAddress,
                "zip_code": zipCode,
                "latitude": latitude,
                "longitude": longitude
            }
            
            # Generate Report Content
            report_content = await generate_report(
                image_content=image_content,
                description=description,
                issue_type=issue_type,
                severity=severity,
                address=streetAddress or "",
                zip_code=zipCode,
                latitude=latitude,
                longitude=longitude,
                issue_id=str(uuid.uuid4()),
                confidence=final_confidence * 100,
                category=category,
                priority=priority
            )
            
            report.status = "waiting_review"
            report.realFlowDetails = RealFlowDetails(
                category=category,
                modelUsed="Expert" if category == "special" else "General",
                kmProcessed=True,
                reportContent=report_content
            )
            
            flow = "real_draft"
            message = "Analysis completed. Please review and submit."
            
        else:
            # Flagged
            report.status = "flagged"
            report.flagFlowDetails = FlagFlowDetails(
                userNotifiedOfFlag=True,
                sentToTeam=True,
                reason="Detected as Fake/Pattern" if not is_real else "Low Confidence/Ambiguous"
            )
            flow = "flagged"
            message = "Image flagged for manual review."
            
        # 5. Save to DB
        db = await get_db()
        report_dict = report.dict(by_alias=True)
        result = await db.reports.insert_one(report_dict)
        report_dict["_id"] = str(result.inserted_id) # Ensure ID is string for return
        
        # Frontend expects { message, flow, report }
        return {
            "message": message,
            "flow": flow,
            "report": report_dict
        }

    except Exception as e:
        logger.error(f"Error in analyze_report: {e}", exc_info=True)
        from fastapi.responses import JSONResponse
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal Server Error",
                "detail": str(e),
                "type": type(e).__name__,
                "trace": traceback.format_exc()
            }
        )

class FinalizeRequest(BaseModel):
    reportId: str

@router.post("/reports/finalize")
async def finalize_report(req: FinalizeRequest):
    try:
        db = await get_db()
        # Find report
        # Need ObjectId? No, we used str(uuid) for _id in model, but Mongo might treat as string if we forced it?
        # Model default factory is uuid string.
        # But wait, insert_one usually creates ObjectId unless we specify _id.
        # In model: id: str = Field(..., alias="_id").
        # If we passed _id in dict, Mongo uses it.
        
        # Let's check how we inserted. We did `report.dict(by_alias=True)`.
        # This includes `_id` as UUID string.
        
        result = await db.reports.update_one(
            {"_id": req.reportId},
            {"$set": {"status": "submitted", "updatedAt": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Report not found")
            
        updated_report = await db.reports.find_one({"_id": req.reportId})
        
        return {
            "message": "Report submitted successfully to authority.",
            "report": updated_report
        }
        
    except Exception as e:
        logger.error(f"Error in finalize_report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
