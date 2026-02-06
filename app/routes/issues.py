from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from services.ai_service import classify_issue
from services.ai_service_optimized import generate_report_optimized as generate_report
from services.email_service import send_email
from services.mongodb_service import store_issue, get_issues, get_user_issues, update_issue_status, get_db, get_fs
from services.geocode_service import reverse_geocode, geocode_zip_code
from services.report_generation_service import build_unified_issue_json
from utils.location import get_authority, get_authority_by_zip_code
from utils.timezone import get_timezone_name
from utils.security import SECRET_KEY, ALGORITHM
from jose import jwt, JWTError
from services.authority_service import resolve_authorities
from bson.objectid import ObjectId
import uuid
import logging
from pathlib import Path
import base64
from datetime import datetime
import pytz
from typing import List, Optional, Dict, Any
import gridfs.errors
import asyncio
from PIL import Image
import io

# Setup optimized logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Set AI service logger to WARNING to reduce noise
logging.getLogger("app.services.ai_service").setLevel(logging.WARNING)
logging.getLogger("app.services.geocode_service").setLevel(logging.WARNING)
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return {"sub": email, "id": payload.get("id"), "role": payload.get("role")}
    except JWTError:
        raise credentials_exception

@router.get("/issues/my-issues")
async def get_my_issues(current_user: dict = Depends(get_current_user)):
    """Get issues reported by the current user."""
    return await get_user_issues(current_user.get("sub"))

class IssueResponse(BaseModel):
    id: str
    message: str
    report: Optional[Dict] = None

class IssueStatusUpdate(BaseModel):
    status: str

class DeclineRequest(BaseModel):
    decline_reason: str
    edited_report: Optional[Dict[str, Any]] = None

class AcceptRequest(BaseModel):
    edited_report: Optional[Dict[str, Any]] = None
    selected_authorities: Optional[List[Dict[str, str]]] = None  # List of {name, email, type}

class SubmitRequest(BaseModel):
    selected_authorities: List[Dict[str, str]]  # List of {name, email, type}
    edited_report: Optional[Dict[str, Any]] = None  # Optional edited report data

class EditedReport(BaseModel):
    issue_overview: Dict[str, Any]

class EmailAuthoritiesRequest(BaseModel):
    issue_id: str
    authorities: List[Dict[str, Any]]  # List of selected authorities
    report_data: Dict[str, Any]  # Report data to include in email
    zip_code: str  # Zip code for context
    recommended_actions: Optional[List[str]] = []  # Make optional with default
    detailed_analysis: Optional[Dict[str, Any]] = {}  # Make optional with default
    responsible_authorities_or_parties: Optional[List[Dict[str, Any]]] = []  # Make optional with default
    template_fields: Optional[Dict[str, Any]] = {}  # Make optional with default

class Issue(BaseModel):
    id: str = Field(..., alias="_id")
    address: str
    zip_code: Optional[str] = None
    latitude: float = 0.0
    longitude: float = 0.0
    issue_type: str
    severity: str
    image_id: str
    status: str = "pending"
    report: Dict = {"message": "No report generated"}
    category: str = "public"
    priority: str = "Medium"
    report_id: str = ""
    timestamp: str
    decline_reason: Optional[str] = None
    decline_history: Optional[List[Dict[str, str]]] = None
    user_email: Optional[str] = None
    authority_email: Optional[List[str]] = None
    authority_name: Optional[List[str]] = None
    timestamp_formatted: Optional[str] = None
    timezone_name: Optional[str] = None
    email_status: Optional[str] = None
    email_errors: Optional[List[str]] = None
    available_authorities: Optional[List[Dict[str, str]]] = None
    recommended_actions: Optional[List[str]] = None
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

def get_logo_base64():
    try:
        logo_path = Path(__file__).parent.parent / "static" / "MomentumAi_4K_Logo-removebg-preview.png"
        if not logo_path.exists():
            logger.error(f"Logo file not found at {logo_path}")
            return None
        with open(logo_path, "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to load logo: {str(e)}", exc_info=True)
        return None

def get_department_email_content(department_type: str, issue_data: dict, is_user_review: bool = False) -> tuple[str, str]:
    issue_type = issue_data.get("issue_type", "Unknown Issue")
    final_address = issue_data.get("address", "Unknown Address")
    zip_code = issue_data.get("zip_code", "Unknown Zip Code")
    timestamp_formatted = issue_data.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
    report = issue_data.get("report", {"message": "No report generated"})
    authority_name = issue_data.get("authority_name", "Department")
    confidence = issue_data.get("confidence", 0.0)
    category = issue_data.get("category", "Public")
    timezone_name = issue_data.get("timezone_name", "UTC")
    latitude = issue_data.get("latitude", 0.0)
    longitude = issue_data.get("longitude", 0.0)
    decline_reason = issue_data.get("decline_reason", "No decline reason provided")
    # Prefer unified JSON fields when available for subject/text
    unified = report.get("unified_report", {}) or issue_data.get("unified_report", {})
    summary_text = (
        unified.get("summary_text")
        or report.get("issue_overview", {}).get("summary_explanation")
        or "No summary available"
    )
    
    severity_checkboxes = {
        "High": "□ High  ☑ Medium  □ Low" if report.get("issue_overview", {}).get("severity", "").lower() == "medium" else "☑ High  □ Medium  □ Low" if report.get("issue_overview", {}).get("severity", "").lower() == "high" else "□ High  □ Medium  ☑ Low",
        "Medium": "□ High  ☑ Medium  □ Low",
        "Low": "□ High  □ Medium  ☑ Low"
    }.get(report.get("issue_overview", {}).get("severity", "Medium").capitalize(), "□ High  ☑ Medium  □ Low")
    
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    
    if is_user_review:
        base_subject = f"Updated Report for {issue_type.title()} at {final_address} - Review Required"
        subject = unified.get("email_subject", base_subject)
        text_content = f"""
Subject: {issue_type.title()} – {final_address} – {timestamp_formatted} – ID {report.get('template_fields', {}).get('oid', 'N/A')}
Dear User,
The report for the {issue_type.title()} issue at {final_address} (Zip: {zip_code}) has been updated based on your feedback: {decline_reason}
Please review the updated report below:
• Issue Type: {category.title()} – {issue_type.title()}
• Time Reported: {timestamp_formatted} {timezone_name}
• Location: {final_address}
• Zip Code: {zip_code}
• GPS: {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}
• Live Location: {map_link}
• Severity: {severity_checkboxes}
• Decline Reason: {decline_reason}
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
 Unified Summary:
 {summary_text}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Please accept the report or provide further feedback by declining with a reason. Reply to this email or contact eaiser@momntumai.com.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
        return subject, text_content
    else:
        templates = {
            "fire": {
                "subject": f"Urgent Fire Hazard Alert – {issue_type.title()} at {final_address}",
                "text_content": f"""
Subject: {issue_type.title()} – {final_address} – {timestamp_formatted} – ID {report.get('template_fields', {}).get('oid', 'N/A')}
Dear {authority_name.title()} Team,
A critical {issue_type.title()} issue has been reported at {final_address} (Zip: {zip_code})
Fire Department Action Required:
• Issue Type: {category.title()} – {issue_type.title()}
• Time Reported: {timestamp_formatted} {timezone_name}
• Location: {final_address}
• Zip Code: {zip_code}
• GPS: {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}
• Live Location: {map_link}
• Severity: {severity_checkboxes}
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact eaiser@momntumai.com for further details.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
            },
            "police": {
                "subject": f"Public Safety Alert – {issue_type.title()} at {final_address}",
                "text_content": f"""
Subject: {issue_type.title()} – {final_address} – {timestamp_formatted} – ID {report.get('template_fields', {}).get('oid', 'N/A')}
Dear {authority_name.title()} Team,
A public safety issue ({issue_type.title()}) has been reported at {final_address} (Zip: {zip_code})
Police Action Required:
• Issue Type: {category.title()} – {issue_type.title()}
• Time Reported: {timestamp_formatted} {timezone_name}
• Location: {final_address}
• Zip Code: {zip_code}
• GPS: {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}
• Live Location: {map_link}
• Severity: {severity_checkboxes}
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact eaiser@momntumai.com for further details.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
            },
            "public_works": {
                "subject": f"Infrastructure Issue – {issue_type.title()} at {final_address}",
                "text_content": f"""
Subject: {issue_type.title()} – {final_address} – {timestamp_formatted} – ID {report.get('template_fields', {}).get('oid', 'N/A')}
Dear {authority_name.title()} Team,
An infrastructure issue ({issue_type.title()}) has been reported at {final_address} (Zip: {zip_code})
Public Works Action Required:
• Issue Type: {category.title()} – {issue_type.title()}
• Time Reported: {timestamp_formatted} {timezone_name}
• Location: {final_address}
• Zip Code: {zip_code}
• GPS: {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}
• Live Location: {map_link}
• Severity: {severity_checkboxes}
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact eaiser@momntumai.com for further details.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
            },
            "general": {
                "subject": f"General Issue – {issue_type.title()} at {final_address}",
                "text_content": f"""
Subject: {issue_type.title()} – {final_address} – {timestamp_formatted} – ID {report.get('template_fields', {}).get('oid', 'N/A')}
Dear {authority_name.title()} Team,
An issue ({issue_type.title()}) has been reported at {final_address} (Zip: {zip_code})
Action Required:
• Issue Type: {category.title()} – {issue_type.title()}
• Time Reported: {timestamp_formatted} {timezone_name}
• Location: {final_address}
• Zip Code: {zip_code}
• GPS: {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}
• Live Location: {map_link}
• Severity: {severity_checkboxes}
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact eaiser@momntumai.com for further details.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
            }
        }
        template = templates.get(department_type, templates["general"])
        # Override subject with unified subject if available, and append summary
        subject = unified.get("email_subject", template["subject"]) 
        text_content = template["text_content"] + f"\nUnified Summary:\n{summary_text}\n"
        return subject, text_content

async def send_authority_email(
    issue_id: str,
    authorities: List[Dict[str, str]],
    issue_type: str,
    final_address: str,
    zip_code: str,
    timestamp_formatted: str,
    report: dict,
    confidence: float,
    category: str,
    timezone_name: str,
    latitude: float,
    longitude: float,
    image_content: bytes,
    decline_reason: Optional[str] = None,
    is_user_review: bool = False
) -> bool:
    if not authorities:
        logger.warning("No authorities provided, using default")
        authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
    
    logo_base64 = get_logo_base64()
    issue_image_base64 = base64.b64encode(image_content).decode('utf-8')
    embedded_images = []
    if logo_base64:
        embedded_images.append(("momentumai_logo", logo_base64, "image/png"))
    embedded_images.append(("issue_image", issue_image_base64, "image/jpeg"))
    
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"

    # --- CLEAN + AUTO SHORT DESCRIPTION (2–3 sentences) ---
    import re

    # Raw description from AI
    full_desc = report.get('issue_overview', {}).get('summary_explanation', 'N/A')

    # Remove unwanted prefixes from AI model output
    clean_desc = re.sub(
        r"(AI Analysis:|\*\*Issue Description:\*\*|\*\*Concise Issue Description:\*\*|\*\*Description:\*\*)",
        "",
        full_desc,
        flags=re.IGNORECASE
    ).strip()

    # Remove all markdown bold markers **text**
    clean_desc = re.sub(r"\*\*", "", clean_desc).strip()

    # Remove multiple spaces
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()

    # Replace double periods
    clean_desc = clean_desc.replace("..", ".")

    # Split into sentences
    sentences = clean_desc.split('.')

    # Keep only first 2–3 sentences
    short_description = '. '.join([s.strip() for s in sentences if s.strip()][:3])

    # Ensure full stop at end
    if not short_description.endswith('.'):
        short_description += '.'


    
    feedback_value = report.get('detailed_analysis', {}).get('feedback')
    feedback_str = str(feedback_value) if feedback_value is not None else 'None'
    
    # Generate enhanced recommended actions HTML
    recommended_actions = report.get('recommended_actions', ['No recommendations provided'])
    recommended_actions_html = ""
    
    for i, action in enumerate(recommended_actions):
        urgency_class = "urgency-immediate" if "immediately" in action.lower() else \
                        "urgency-high" if "urgent" in action.lower() or "24 hours" in action.lower() else \
                        "urgency-medium" if "48 hours" in action.lower() else "urgency-low"
        
        urgency_text = "Immediate" if "immediately" in action.lower() else \
                       "High" if "urgent" in action.lower() or "24 hours" in action.lower() else \
                       "Medium" if "48 hours" in action.lower() else "Standard"
        
        recommended_actions_html += f"""
        <div class="action-item">
            <div class="action-icon">{i+1}</div>
            <div class="action-text">
                {action}
                <span class="action-urgency {urgency_class}">
                    {urgency_text}
                </span>
            </div>
        </div>
        """
    
    # Compute severity label for template usage
    severity_checkboxes = (
        str(report.get('issue_overview', {}).get('severity') or report.get('template_fields', {}).get('priority') or 'Medium')
    )
    severity_checkboxes = severity_checkboxes.title()

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>

    /* ---------------------------------------------------------
       UNIVERSAL ANIMATIONS
    --------------------------------------------------------- */

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}

    @keyframes slideIn {{
        0%   {{ transform: translateY(25px); opacity: 0; }}
        100% {{ transform: translateY(0); opacity: 1; }}
    }}

    @keyframes shimmerMove {{
        0%   {{ background-position: 0% 0%; }}
        100% {{ background-position: 200% 0%; }}
    }}

    @keyframes glowPulse {{
        0%   {{ box-shadow: 0 0 4px rgba(255,204,51,0.3); }}
        50%  {{ box-shadow: 0 0 16px rgba(255,204,51,1); }}
        100% {{ box-shadow: 0 0 4px rgba(255,204,51,0.3); }}
    }}

    @keyframes fillBar {{
        from {{ width: 0%; }}
        to   {{ width: {confidence}%; }}
    }}

    /* ---------------------------------------------------------
       PAGE BASE
    --------------------------------------------------------- */

    body {{
        background: #0b0b0f;
        font-family: Segoe UI, Arial, sans-serif;
        padding: 25px;
        color: #f6f6f6;
        animation: fadeIn 0.8s ease-out;
    }}

    .container {{
        max-width: 720px;
        margin: auto;
        background: #121217;
        border-radius: 12px;
        padding: 30px;
        border: 1px solid #1f1f26;
        animation: slideIn 0.9s ease-out;
    }}

    /* ---------------------------------------------------------
       SECTION TITLE
    --------------------------------------------------------- */

    .section-title {{
        font-size: 20px;
        font-weight: 700;
        color: #f6c521;
        margin-bottom: 12px;
        text-shadow: 0 0 8px rgba(246,197,33,0.4);
    }}

    .label {{ font-weight: 600; color:#cfcfcf; }}
    .value {{ color:#fefefe; }}

    /* ---------------------------------------------------------
       CONFIDENCE BAR (Email Safe)
    --------------------------------------------------------- */

    .conf-container {{
        width: 100%;
        height: 14px;
        background: #1c1c23;
        border: 1px solid #292933;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 6px;
        position: relative;
    }}

    .conf-fill {{
        height: 100%;
        width: 0%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ffcc33, #ffdf72, #ffcc33);
        animation:
            fillBar 2s ease-out forwards,
            shimmerMove 2s linear infinite;
        background-size: 200% 100%;
        position: relative;
    }}

    .conf-fill::after {{
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 10px;
        animation: glowPulse 2s infinite ease-in-out;
    }}

    /* SCANLINES */
    .scanlines {{
        width: 100%;
        height: 5px;
        border-radius: 4px;
        margin-top: 4px;
        background: linear-gradient(90deg, transparent, rgba(246,197,33,0.18), transparent);
        background-size: 200% 100%;
        animation: shimmerMove 3s linear infinite;
    }}

    /* ---------------------------------------------------------
       IMAGE
    --------------------------------------------------------- */

    .issue-image {{
        width: 100%;
        margin-top: 10px;
        border-radius: 8px;
        border: 1px solid #2d2d33;
    }}

    hr {{
        border: none;
        border-bottom: 1px solid #23232a;
        margin: 25px 0;
    }}

</style>

</head>
<body>

<div class="container">

    <!-- HEADER -->
    <h1 style="color:#f6c521; margin:0 0 5px 0;">EAiSER CIVIC – Incident Report</h1>
    <p style="color:#8f8f8f; margin-top:0;">Automated Issue Analysis & Routing System</p>
    <hr>

    <!-- 1. SUMMARY -->
    <div>
        <div class="section-title">1. Incident Summary</div>

        <p><span class="label">Issue:</span> <span class="value">{issue_type}</span></p>

        <p><span class="label">Description:</span><br>
            <span class="value">{short_description}</span>
        </p>

        <p><span class="label">Location:</span> <span class="value">{final_address}</span>
            — <a href="{map_link}" style="color:#f6c521;">View Map</a></p>

        <p><span class="label">Priority:</span>
            {report.get('template_fields', {}).get('priority')}</p>

        <p><span class="label">Reported:</span> {timestamp_formatted} {timezone_name}</p>
    </div>

    <hr>

    <!-- 2. IMAGE -->
    <div>
        <div class="section-title">2. Image Evidence</div>
        <img src="cid:issue_image" class="issue-image">
    </div>

    <hr>

    <!-- 3. AUTHORITY -->
    <div>
        <div class="section-title">3. Recommended Authority</div>
        <p><span class="label">Route To:</span>
            {", ".join([auth["name"] for auth in authorities])}
        </p>
    </div>

    <hr>

    <!-- 4. SYSTEM METADATA -->
    <div>
        <div class="section-title">4. System Metadata</div>

        <p><span class="label">Report ID:</span>
            {report.get('template_fields', {}).get('oid', 'N/A')}</p>

        <p><span class="label">AI Confidence:</span> {confidence}%</p>

        <div class="conf-container">
            <div class="conf-fill"></div>
        </div>

        <div class="scanlines"></div>

        <p style="margin-top:15px;">
            <span class="label">Coordinates:</span> {latitude}, {longitude}
        </p>
    </div>

    <hr>

    <!-- FOOTER -->
    <p style="font-size:12px; color:#888;">
        This report was generated automatically by EAiSER AI.<br>
        © 2025 MomntumAi LLC — All Rights Reserved.
    </p>

</div>

</body>
</html>

"""
    city_val = final_address.split(',')[0] if final_address else "Unknown"
    state_val = "Unknown"
    try:
        parts = [p.strip() for p in final_address.split(',')] if final_address else []
        if len(parts) >= 2:
            state_val = parts[1]
    except Exception:
        state_val = "Unknown"
    confidence_percent = int(confidence) if isinstance(confidence, (int, float)) else 0
    image_name = report.get('template_fields', {}).get('image_filename', 'N/A')
    impact_desc = report.get('detailed_analysis', {}).get('potential_impact', 'N/A')
    location_ctx = report.get('issue_overview', {}).get('location_context', 'N/A')
    report_oid = report.get('template_fields', {}).get('oid', 'N/A')
    
    # Calculate color for confidence bar
    conf_val = float(confidence) if isinstance(confidence, (int, float, str)) and str(confidence).replace('.','',1).isdigit() else 0
    if conf_val >= 80:
        conf_bar_color = "#4ade80" # Green
    elif conf_val >= 50:
        conf_bar_color = "#facc15" # Yellow
    else:
        conf_bar_color = "#ef4444" # Red
        
    subject_override = f"EAiSER Alert – {issue_type} (ID: {report_oid})"
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>EAiSER CIVIC – Incident Report</title>

<style>
  .container {{
    width: 100%;
    max-width: 720px;
    margin: 0 auto;
    padding: 20px;
    background: #0b0b0f;
    color: #f6f6f6;
    font-family: Segoe UI, Arial, sans-serif;
  }}
  .card {{
    background: #121217;
    border: 1px solid #1f1f26;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 25px;
  }}
  h1, h2 {{
    margin: 0 0 10px;
    color: #f6c521;
  }}
  a {{
    color: #f6c521;
    text-decoration: none;
  }}
  .label {{
    color: #dcdcdc;
    font-weight: 600;
  }}
  hr {{
    border: 0;
    border-bottom: 1px solid #222;
    margin: 20px 0;
  }}

  /* Confidence bar */
  .conf-bar-container {{
    background: #1b1b23;
    border: 1px solid #2a2a33;
    height: 14px;
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
  }}
  .conf-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #f6c521, #ffdf6a);
    animation: loadConf 1.8s ease-out forwards;
    width: 0%;
  }}
  @keyframes loadConf {{
    0% {{ width: 0%; }}
    100% {{ width: {confidence}%; }}
  }}
</style>

</head>

<body style="margin:0;padding:0;">
<table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#0b0b0f;width:100%;margin:0;padding:0;">
<tr>
<td align="center" style="background:#0b0b0f;padding:20px;">

<div class="container">

  <h1>EAiSER CIVIC – Incident Report</h1>
  <p style="color:#888;font-size:14px;margin-top:-6px;">
    Automated Issue Analysis & Routing System
  </p>
  <hr />

  <!-- INCIDENT SUMMARY -->
  <div class="card">
    <h2>1. Incident Summary</h2>

    <p><span class="label">Issue:</span> {issue_type}</p>

    <p><span class="label">Description:</span>
       {short_description}
    <span style="color:#777;font-size:12px;">(Auto-summary)</span>
    </p>

    <p><span class="label">Location:</span> {final_address}
      <a href="{map_link}" target="_blank"> — View Map</a>
    </p>

    <p><span class="label">Priority:</span>
       {report.get('issue_overview', {}).get('severity_label', 'N/A')}
    </p>

    <p><span class="label">Reported:</span> {timestamp_formatted}</p>
  </div>

  <!-- IMAGE -->
  <div class="card">
    <h2>2. Image Evidence</h2>
    <p style="color:#ccc;">Attached below:</p>
    <img src="cid:issue_image"
         style="max-width:100%;border-radius:6px;border:1px solid #2a2a33;" />
  </div>

  <!-- AUTHORITY -->
  <div class="card">
    <h2>3. Recommended Authority</h2>
    <p><span class="label">Route To:</span>
      {", ".join([auth["name"] for auth in authorities])}
    </p>
  </div>

  <!-- SYSTEM METADATA -->
  <div class="card">
    <h2>4. System Metadata</h2>

    <p><span class="label">Report ID:</span>
      {report.get('template_fields', {}).get('oid', 'N/A')}
    </p>

    <p><span class="label">AI Confidence:</span> {confidence}%</p>

    <div class="conf-bar-container">
      <div class="conf-bar-fill" style="width: {confidence}%; background: {conf_bar_color};"></div>
    </div>

    <br />

    <p><span class="label">Coordinates:</span>
       {latitude}, {longitude}
    </p>
  </div>

  <hr />

  <!-- FOOTER -->
  <p style="font-size:12px;color:#888;line-height:1.6;">
    This report was submitted through <strong style="color:#f6c521;">EAiSER CIVIC</strong>, powered by MomntumAi LLC.<br>
    EAiSER analyzes images, classifies incidents, assigns priority and routes issues to relevant authorities.
  </p>

  <p style="font-size:11px;color:#555;">© 2025 MomntumAi LLC — All Rights Reserved.</p>

</div>

</td>
</tr>
</table>

</body>
</html>

"""
    errors = []
    successful_emails = []
    
    for authority in authorities:
        try:
            subject, text_content = get_department_email_content(
                authority.get("type", "general"),
                {
                    "issue_type": issue_type,
                    "address": final_address,
                    "zip_code": zip_code,
                    "timestamp_formatted": timestamp_formatted,
                    "report": report,
                    "authority_name": authority.get("name", "Department"),
                    "confidence": confidence,
                    "category": category,
                    "timezone_name": timezone_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "decline_reason": decline_reason
                },
                is_user_review=is_user_review
            )
            logger.debug(f"Sending email to [redacted] for {authority.get('type', 'general')} with subject: {subject}")
            success = await send_email(
                to_email=authority.get("email", "eaiser@momntumai.com"),
                subject=subject_override or subject,
                html_content=html_content,
                text_content=text_content,
                attachments=None,
                embedded_images=embedded_images
            )
            if success:
                successful_emails.append(authority.get("email", "eaiser@momntumai.com"))
                logger.info(f"Email sent successfully to [redacted] for {authority.get('type', 'general')}")
            else:
                logger.warning(f"Email sending failed for [redacted] without raising an exception")
                errors.append(f"Email sending failed for {authority.get('email', 'eaiser@momntumai.com')}")
        except Exception as e:
            logger.error(f"Failed to send email to [redacted]: {str(e)}", exc_info=True)
            errors.append(f"Failed to send email to {authority.get('email', 'eaiser@momntumai.com')}: {str(e)}")
    
    try:
        db = await get_db()
        await db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "email_status": "sent" if successful_emails else "failed",
                    "email_errors": errors
                }
            }
        )
        logger.debug(f"Issue {issue_id} updated with email_status: {'sent' if successful_emails else 'failed'}")
    except Exception as e:
        logger.error(f"Failed to log email attempt for issue {issue_id}: {str(e)}", exc_info=True)
        errors.append(f"Failed to log email attempt: {str(e)}")
    
    if errors:
        logger.warning(f"Email sending issues for issue {issue_id}: {'; '.join(errors)}")
    if successful_emails:
        logger.info(f"Emails sent successfully for issue {issue_id} to: {', '.join(successful_emails)}")
    
    return len(errors) == 0

@router.post("/issues", response_model=IssueResponse)
async def create_issue(
    image: Optional[UploadFile] = File(None),  # Image is now OPTIONAL
    description: str = Form(''),
    address: str = Form(''),
    zip_code: Optional[str] = Form(None),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    user_email: Optional[str] = Form(None),
    category: str = Form('public'),
    
    severity: str = Form('medium'),
    issue_type: str = Form('other')
):
    logger.debug(f"Creating issue with address: {address}, zip: {zip_code}, lat: {latitude}, lon: {longitude}")
    try:
        db = await get_db()
        fs = await get_fs()
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    # -------------------------------------------------------------
    # 1. MANUAL REPORT FLOW (No Image)
    # -------------------------------------------------------------
    if image is None:
        logger.info("📝 Processing MANUAL REPORT (No Image Provided)")
        
        issue_id = str(uuid.uuid4())
        
        # Enforce "Unknown/Manual" state so user must fill it
        final_address = address
        if not final_address and latitude and longitude:
             try:
                geocode_result = await reverse_geocode(latitude, longitude)
                final_address = geocode_result.get("address", "Unknown Address")
                if not zip_code:
                     zip_code = geocode_result.get("zip_code")
             except:
                final_address = "Unknown Address"

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Construct a "Blank" Report for Manual Entry
        manual_report = {
            "issue_overview": {
                "issue_type": issue_type if issue_type != 'other' else "Manual Report",
                "category": category,
                "severity": severity or "medium",
                "summary_explanation": description or "Manual report submitted by user.",
                "confidence": 0,  # 0 Confidence -> Triggers "Manual Mode" in UI
                "location_context": "Manual Entry"
            },
            "detailed_analysis": {
                "root_causes": "Manual Report",
                "public_safety_risk": "Unknown",
                "potential_consequences_if_ignored": "Unknown",
                "environmental_impact": "Unknown",
                "structural_implications": "Unknown",
                "legal_or_regulatory_considerations": "None",
                "feedback": "None"
            },
            "recommended_actions": ["Review Details", "Assign Authority"],
            "responsible_authorities_or_parties": [], # Empty -> User must select
            "available_authorities": [],
            "ai_evaluation": {
                "issue_detected": True,
                "detected_issue_type": "Manual",
                "ai_confidence_percent": 0,
                "image_analysis": "No image provided.",
                "rationale": "User opted for manual reporting."
            },
            "template_fields": {
                "oid": issue_id,
                "timestamp": timestamp_str,
                "confidence": 0,
                "ai_tag": "Manual",
                "address": final_address,
                "priority": "Medium",
                "image_filename": "No Image"
            }
        }
        
        # We DO NOT save to DB yet. We return it as a "Preview" for the user to edit/confirm.
        # But we need an ID to submit later.
        # Ideally, we save a draft or just user sends it back. 
        # The frontend expects 'id' and 'report'.
        # We will save a placeholder 'pending_manual' issue to allow 'submit' to work?
        # Actually 'submit_issue' expects an existing issue in DB.
        # So we MUST save this initial shell to DB.
        
        # Save to DB (Status: draft/pending)
        await store_issue(
             db,
             fs,
             issue_id,
             b'', # Empty image content
             manual_report,
             {}, # unified_report
             final_address,
             zip_code,
             latitude,
             longitude,
             manual_report['issue_overview']['issue_type'],
             manual_report['issue_overview']['severity'],
             "General", # category
             "Medium", # priority
             user_email,
             [], # responsible_authorities
             [] # available_authorities
        )
        
        return IssueResponse(
            id=issue_id,
            message="Manual draft created. Please fill in details.",
            report={
                 "report": manual_report,
                 "id": issue_id
            }
        )

    # -------------------------------------------------------------
    # 2. STANDARD AI FLOW (With Image)
    # -------------------------------------------------------------
    if not image.content_type.startswith("image/"):
        logger.error(f"Invalid image format: {image.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        image_content = await image.read()
        logger.debug(f"Image read successfully, size: {len(image_content)} bytes")
        
        # Performance: Resize/Optimize image before processing
        try:
            with Image.open(io.BytesIO(image_content)) as img:
                # Resize if larger than 1024x1024
                if img.width > 1024 or img.height > 1024:
                    img.thumbnail((1024, 1024))
                
                # Convert to RGB (standardize)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compress to JPEG
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85, optimize=True)
                new_content = output.getvalue()
                
                # Only use new content if it's actually smaller or we resized
                if len(new_content) < len(image_content):
                    image_content = new_content
                    logger.info(f"⚡ Image optimized: {len(image_content)} bytes")
        except Exception as e:
            logger.warning(f"⚠️ Image optimization failed (using original): {e}")

    except Exception as e:
        logger.error(f"Failed to read image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")
    
    try:
        # Pass reduced image to initial classifier
        issue_type, severity, confidence, category, priority = await classify_issue(image_content, description or "")
        if not issue_type:
            logger.error("Failed to classify issue type")
            raise ValueError("Failed to classify issue type")
        logger.debug(f"Issue classified: type={issue_type}, severity={severity}, confidence={confidence}, category={category}, priority={priority}")
    except Exception as e:
        logger.error(f"Failed to classify issue: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to classify issue: {str(e)}")
    
    final_address = address
    if zip_code:
        try:
            geocode_result = await geocode_zip_code(zip_code)
            final_address = geocode_result.get("address", address or "Unknown Address")
            latitude = geocode_result.get("latitude", latitude)
            longitude = geocode_result.get("longitude", longitude)
            logger.debug(f"Geocoded zip code {zip_code}: address={final_address}, lat={latitude}, lon={longitude}")
        except Exception as e:
            logger.warning(f"Failed to geocode zip code {zip_code}: {str(e)}", exc_info=True)
            final_address = address or "Unknown Address"
    elif not address and latitude and longitude:
        try:
            geocode_result = await reverse_geocode(latitude, longitude)
            final_address = geocode_result.get("address", "Unknown Address")
            zip_code = geocode_result.get("zip_code", zip_code)
            logger.debug(f"Geocoded address: {final_address}, zip: {zip_code}")
        except Exception as e:
            logger.warning(f"Failed to geocode coordinates ({latitude}, {longitude}): {str(e)}", exc_info=True)
            final_address = "Unknown Address"
    
    issue_id = str(uuid.uuid4())
    try:
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
        report["template_fields"].pop("tracking_link", None)
        # Preserve enriched address from AI if present; otherwise fallback to final_address
        tf = report.get("template_fields", {})
        tf["zip_code"] = zip_code or "N/A"
        ai_addr = (tf.get("address") or "").strip()
        if not ai_addr or ai_addr.lower() in {"unknown address", "not specified", "n/a", ""}:
            tf["address"] = final_address or "Not specified"
        report["template_fields"] = tf
        
        # --- VALIDATE REPORT QUALITY (TRIGGER ALERTS) ---
        ai_eval = report.get("ai_evaluation", {})
        issue_overview = report.get("issue_overview", {})
        
        # 1. Check for Fake/Simulated Image
        # The AI, in 'generate_report_optimized', sets 'issue_detected'=False and type='None' for fake images
        is_issue_detected = ai_eval.get("issue_detected")
        detected_type = str(issue_overview.get("type", "")).lower()
        confidence_val = issue_overview.get("confidence", 0)
        
        # Conditions for rejection:
        # A. Explicitly detected as fake (confidence usually forced to ~5)
        if "fake" in str(ai_eval.get("image_analysis", "")).lower() or \
           "simulated" in str(ai_eval.get("rationale", "")).lower():
            logger.warning(f"🚫 Rejected fake image analysis for issue {issue_id}")
            raise HTTPException(status_code=400, detail="Analysis failed: This image appears to be AI-generated or manipulated.")

        # B. Confidence too low (Blurry or Irrelevant)
        if confidence_val < 5: 
            logger.warning(f"🚫 Rejected extremely low confidence report ({confidence_val}%) for issue {issue_id}")
            raise HTTPException(status_code=400, detail="Analysis failed: The image is too blurry, dark, or unclear to analyze.")
            
        # C. No issue detected
        logger.info(f"Validation Check: type='{detected_type}', detected={is_issue_detected}, conf={confidence_val}")
        
        # C. No issue detected - Relaxed check (< 30 instead of < 45)
        if is_issue_detected is False and confidence_val < 30:
             # Allow "Other" if confidence is decent, but reject "None"
             if detected_type in ["none", ""]:
                 raise HTTPException(status_code=400, detail="Analysis failed: No valid infrastructure issue detected in this image.")

        recommended_actions = report.get("recommended_actions", [])
        if "recommended_actions" not in report:
            report["recommended_actions"] = recommended_actions
        
        logger.debug(f"Report generated for issue {issue_id}")
    except HTTPException:
        raise # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Failed to generate report for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    try:
        # Resolve authorities using the new service
        # Extract context from the generated report
        ai_json_context = report.get("issue_overview", {})
        if not ai_json_context:
             # Fallback context if report structure is different
             ai_json_context = {
                 "case_id": issue_id,
                 "description": description,
                 "confidence": confidence,
                 "summary_explanation": description
             }
        else:
             ai_json_context["case_id"] = issue_id
             ai_json_context["confidence"] = confidence

        authority_result = await resolve_authorities(issue_type, zip_code, ai_json_context)
        
        resolved_auths = authority_result["authorities"]
        
        # Consolidate to responsible_authorities
        responsible_authorities = []
        if resolved_auths:
            responsible_authorities = [
                {**{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}, **auth}
                for auth in resolved_auths
            ]
        else:
             logger.warning(f"No authorities resolved for {issue_type} in {zip_code}")
             responsible_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]

        available_authorities = responsible_authorities

        authority_emails = [auth["email"] for auth in responsible_authorities]
        authority_names = [auth["name"] for auth in responsible_authorities]
        
        logger.info(f"Authorities resolved: {len(responsible_authorities)} found. Mapped: {authority_result.get('is_mapped')}")
        if not authority_result.get('is_mapped'):
             logger.info(f"⚠️ Issue unmapped. Review entry created: {authority_result.get('mapping_review', {}).get('id')}")

    except Exception as e:
        logger.warning(f"Failed to fetch authorities: {str(e)}. Using default authority.", exc_info=True)
        responsible_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
        available_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
        authority_emails = ["eaiser@momntumai.com"]
        authority_names = ["City Department"]
    
    timezone_name = get_timezone_name(latitude, longitude) or "UTC"
    timestamp = datetime.utcnow().isoformat()
    timestamp_formatted = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        final_zip_code = zip_code if zip_code else "N/A"
        final_address = final_address if final_address else "Unknown Address"
        final_latitude = latitude if latitude else 0.0
        final_longitude = longitude if longitude else 0.0
        # Build unified report JSON for consistent UI/Email rendering
        confidence_val = 0.0
        try:
            conf_candidates = [
                report.get("template_fields", {}).get("confidence"),
                report.get("unified_report", {}).get("confidence"),
                report.get("issue_overview", {}).get("confidence"),
            ]
            for c in conf_candidates:
                if c is None:
                    continue
                s = str(c).strip()
                if s.endswith('%'):
                    s = s[:-1]
                v = float(s)
                if v <= 1.0:
                    v = v * 100.0
                confidence_val = max(0.0, min(100.0, v))
                break
        except Exception:
            confidence_val = 0.0
        unified_report = build_unified_issue_json(
            report=report,
            issue_id=issue_id,
            issue_type=issue_type,
            category=category,
            severity=severity,
            priority=priority,
            confidence=confidence_val,
            address=final_address,
            zip_code=final_zip_code,
            latitude=final_latitude,
            longitude=final_longitude,
            timestamp_formatted=timestamp_formatted,
            timezone_name=timezone_name,
            department_type=None,
            is_user_review=False,
        )
        # Also attach unified report inside the report dict for downstream email rendering
        try:
            report["unified_report"] = unified_report
            report["responsible_authorities_or_parties"] = responsible_authorities
            report["available_authorities"] = available_authorities
        except Exception:
            pass
        image_id = await store_issue(
            db=db,
            fs=fs,
            issue_id=issue_id,
            image_content=image_content,
            report=report,
            unified_report=unified_report,
            address=final_address,
            zip_code=final_zip_code,
            latitude=final_latitude,
            longitude=final_longitude,
            issue_type=issue_type,
            severity=severity,
            category=category,
            priority=priority,
            user_email=user_email,
            responsible_authorities=report["responsible_authorities_or_parties"],
            available_authorities=report["available_authorities"]
        )
        logger.info(f"Issue {issue_id} stored successfully with image_id {image_id}")
    except Exception as e:
        logger.error(f"Failed to store issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store issue: {str(e)}")
    
    try:
        db = await get_db()
        db.issues.update_one(
            {"_id": issue_id},
            {"$set": {"recommended_actions": recommended_actions}}
        )
        logger.debug(f"Added recommended_actions to issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to store issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store issue: {str(e)}")

    # Safety guard: evaluate dispatch decision and screen out benign/prank
    try:
        from app.services.dispatch_guard_service import AuthorityDispatchGuard
    except ImportError:
        from services.dispatch_guard_service import AuthorityDispatchGuard

    try:
        overview = report.get("issue_overview", {})
        desc_text = str(overview.get("summary_explanation", "")).lower()
        labels_list = overview.get("detected_problems", [])
        labels_text = " ".join([str(x).lower() for x in labels_list])
        combined = f"{desc_text} {labels_text}"
        severity_val = str(overview.get("severity", severity or "medium")).lower()
        confidence_val = 0.0
        try:
            confidence_val = float(report.get("template_fields", {}).get("confidence", 0) or 0)
        except Exception:
            confidence_val = 0.0

        tokens_controlled_fire = [
            "campfire", "bonfire", "bon fire", "bbq", "barbecue", "barbeque", "grill", "fire pit", "controlled burn",
            "festival", "celebration", "diwali", "diya", "candle", "incense", "lamp", "stove", "kitchen", "smoke machine", "stage"
        ]
        tokens_fire = ["fire", "smoke", "flame", "burning", "wildfire", "house fire", "building fire"]
        danger_words = ["danger", "hazard", "out of control", "emergency", "injury", "uncontrolled", "explosion"]
        is_controlled_fire = any(w in combined for w in tokens_controlled_fire)
        has_fire = any(w in combined for w in tokens_fire)
        is_danger = any(w in combined for w in danger_words)

        policy_conflict = bool(has_fire and is_controlled_fire and not is_danger and severity_val in ["low", "medium"])  # benign
        metadata_ok = bool(final_address) and (bool(final_zip_code) or (final_latitude and final_longitude))

        guard = AuthorityDispatchGuard()
        decision = guard.evaluate(
            {
                "severity": severity_val,
                "priority": str(priority or "medium").lower(),
                "ai_confidence_percent": confidence_val,
                "metadata_complete": metadata_ok,
                "is_duplicate": False,
                "policy_conflict": policy_conflict,
            }
        )

        issue_status = "pending"
        # If action is explicitly reject, screen out.
        if decision.action == "reject":
            issue_status = "screened_out"
            logger.warning(f"⚠️ Issue {issue_id} SCREENED OUT by dispatch guard")
        # If action is 'route_to_review_team', we keep it pending (or move to a specific 'needs_review' status if supported)
        # but importantly, we do NOT screen it out. 
        # For now, "pending" allows it to be seen in the admin dashboard for review.
        elif decision.action == "route_to_review_team":
            issue_status = "needs_review"  # Ensure it is visible for review IMMEDIATELY
            logger.info(f"✅ Issue {issue_id} set to NEEDS_REVIEW (confidence={confidence_val}%, type={issue_type})")
            # Optionally add a flag or note in dispatch_reasons (already handled by decision.reasons)
        else:
            logger.info(f"📝 Issue {issue_id} set to PENDING (action={decision.action})")

        db = await get_db()
        db.issues.update_one(
            {"_id": issue_id},
            {"$set": {
                "dispatch_decision": decision.action,
                "dispatch_reasons": decision.reasons,
                "risk_score": decision.risk_score,
                "fraud_score": decision.fraud_score,
                "status": issue_status,
            }}
        )
        
        logger.info(f"🔄 Issue {issue_id} saved with status={issue_status}")

        try:
            report.setdefault("unified_report", {}).setdefault("dispatch_decision", {})
            report["unified_report"]["dispatch_decision"] = {
                "action": decision.action,
                "reasons": decision.reasons,
                "risk_score": decision.risk_score,
                "fraud_score": decision.fraud_score,
                "status": issue_status,
            }
        except Exception:
            pass
    except Exception:
        pass
    
    try:
        user_authority = [{"name": "User", "email": user_email or "eaiser@momntumai.com", "type": "general"}]
        email_success = await send_authority_email(
            issue_id=issue_id,
            authorities=user_authority,
            issue_type=issue_type,
            final_address=final_address,
            zip_code=zip_code or "N/A",
            timestamp_formatted=timestamp_formatted,
            report=report,
            confidence=confidence,
            category=category,
            timezone_name=timezone_name,
            latitude=latitude,
            longitude=longitude,
            image_content=image_content,
            is_user_review=True
        )
        db = await get_db()
        db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "email_status": "sent" if email_success else "failed",
                    "email_errors": [] if email_success else ["Failed to send initial review email"]
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to send initial review email for issue {issue_id}: {str(e)}", exc_info=True)
    
    return IssueResponse(
        id=issue_id,
        message="Please review the generated report and select responsible authorities",
        report={
            "issue_id": issue_id,
            "status": issue_status,
            "dispatch_decision": decision.action,
            "report": report,
            "authority_email": authority_emails,
            "authority_name": authority_names,
            "available_authorities": available_authorities,
            "recommended_actions": recommended_actions,
            "timestamp_formatted": timestamp_formatted,
            "timezone_name": timezone_name,
            "image_content": base64.b64encode(image_content).decode('utf-8')
        }
    )

@router.post("/issues/{issue_id}/submit", response_model=IssueResponse)
async def submit_issue(issue_id: str, request: SubmitRequest):
    logger.debug(f"Processing submit request for issue {issue_id}")
    try:
        db = await get_db()
        fs = await get_fs()
        logger.debug("Database and GridFS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    try:
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
            logger.error(f"Issue {issue_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        if issue.get("status") == "screened_out":
            logger.warning(f"Issue {issue_id} was screened out but user is submitting; routing to Admin Review.")
            # Do NOT block. Allow to proceed so it can be flagged as needs_review below.
        elif issue.get("status") and issue.get("status") not in ["pending", "needs_review"]:
            logger.warning(f"Issue {issue_id} already processed with status {issue.get('status')}")
            raise HTTPException(status_code=400, detail="Issue already processed")
    except Exception as e:
        logger.error(f"Failed to fetch issue {issue_id} from database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch issue: {str(e)}")
    
    required_fields = ["issue_type", "address", "image_id", "report"]
    missing_fields = [field for field in required_fields if field not in issue or issue[field] is None]
    if missing_fields:
        logger.error(f"Issue {issue_id} missing required fields: {missing_fields}")
        raise HTTPException(status_code=400, detail=f"Issue missing required fields: {missing_fields}")
    
    selected_authorities = request.selected_authorities
    if not selected_authorities:
        logger.error(f"No authorities selected for issue {issue_id}")
        raise HTTPException(status_code=400, detail="At least one authority must be selected")
    
    for auth in selected_authorities:
        if not all(key in auth for key in ["name", "email", "type"]):
            logger.error(f"Invalid authority format for issue {issue_id}: {auth}")
            raise HTTPException(status_code=400, detail="Each authority must have name, email, and type")
        if not auth["email"].endswith("@momntumai.com") and not any(auth["email"] == avail["email"] for avail in issue.get("available_authorities", [])):
            logger.warning(f"Custom authority email {auth['email']} not in available authorities for issue {issue_id}")
            auth["type"] = auth.get("type", "custom")
    
    try:
        # Get image content from GridFS
        gridout = await fs.open_download_stream(ObjectId(issue["image_id"]))
        image_content = await gridout.read()
        logger.debug(f"Image {issue['image_id']} retrieved for issue {issue_id}")
    except gridfs.errors.NoFile:
        logger.error(f"Image not found for image_id {issue['image_id']} in issue {issue_id}")
        raise HTTPException(status_code=404, detail=f"Image not found for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to fetch image for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(e)}")
    
    report = issue["report"]

    # 1. Apply Edits to Report Object (InMemory) - BEFORE Guard Logic
    if request.edited_report:
        logger.debug(f"Updating report with edited content for issue {issue_id}")
        # Merge edited report data into existing report
        for key, value in request.edited_report.items():
            if key in report and isinstance(report[key], dict) and isinstance(value, dict):
                report[key].update(value)
            else:
                report[key] = value

    # Enforce guard decision: block low confidence or policy conflicts
    # FAIL SAFE LOGIC
    # Calculate confidence first to guide decision
    conf_val = 0.0
    try:
        # Try to get confidence from multiple sources in order of preference
        conf_candidates = [
            request.edited_report.get('issue_overview', {}).get('confidence') if request.edited_report else None,
            report.get("template_fields", {}).get("confidence"),
            report.get("unified_report", {}).get("confidence"),
            report.get("issue_overview", {}).get("confidence"),
            issue.get("report", {}).get("issue_overview", {}).get("confidence")
        ]
        
        # Collect all valid confidence scores
        valid_scores = []
        for c in conf_candidates:
            if c is None: continue
            try:
                s = str(c).strip()
                if s.endswith('%'): s = s[:-1]
                v = float(s)
                if v <= 1.0: v = v * 100.0
                v = max(0.0, min(100.0, v))
                valid_scores.append(v)
            except Exception: continue
        
        # Use the highest found confidence to be permissive, or min if we want to be conservative? 
        # User wants High Confidence -> Auto Send. Let's use the explicit value found.
        # Usually these should align. let's take the first valid one we found (priority order).
        if valid_scores:
            conf_val = valid_scores[0] # Priority: Edited > Template > Unified > Overview
        else:
            conf_val = 0.0

    except Exception as e:
        logger.error(f"DEBUG: Confidence parsing error: {e}")
        conf_val = 0.0

    # Determine Issue Type
    current_issue_type = report.get("issue_type") or issue.get("issue_type", "unknown")
    current_issue_type = str(current_issue_type).lower().strip()

    # Restricted Categories that ALWAYS require review
    restricted_categories = [
        "other", "none", "unknown",
        "controlled_fire", "bonfire", "campfire", "burning_leaves",
        "festival", "ceremony", "bbq", "barbecue",
        "fake", "ai_generated", "cartoon", "animated", "illustration", "art", "drawing", "video_game"
    ]
    
    # Check for restricted types (Exact match or Safe substring)
    # We explicitly exclude "uncontrolled" from matching "controlled" logic
    is_restricted = False
    if current_issue_type in restricted_categories:
        is_restricted = True
    else:
        # Substring checks
        # Fire/Controlled checks
        if "bonfire" in current_issue_type or "campfire" in current_issue_type:
            is_restricted = True
        elif "control" in current_issue_type and "uncontrolled" not in current_issue_type:
            is_restricted = True
        
        # Artificial/Fake checks
        artificial_terms = ["fake", "ai generated", "cartoon", "animated", "illustration", "drawing"]
        if any(term in current_issue_type for term in artificial_terms):
            is_restricted = True

    # DECISION LOGIC
    # 1. If restricted category -> Review
    # 2. If confidence < 70 -> Review
    # 3. Otherwise -> Auto Send
    
    if is_restricted:
        logger.info(f"🚨 Issue {issue_id} flagged for review (Restricted Category: '{current_issue_type}')")
        needs_review = True
    elif conf_val < 70:
        logger.info(f"🚨 Issue {issue_id} flagged for review (Low Confidence: {conf_val}%)")
        needs_review = True
    else:
        logger.info(f"✅ Issue {issue_id} passed review checks (Conf={conf_val}%, Type={current_issue_type}). Auto-sending.")
        needs_review = False

    if needs_review:
        # Save state and return
        try:
             # Ensure recommended actions if missing
            recommended_actions = report.get("recommended_actions", [])
            
            await db.issues.update_one(
                {"_id": issue_id},
                {
                    "$set": {
                        "report": report,
                        "status": "needs_review",
                        "recommended_actions": recommended_actions,
                        "authority_email": [auth["email"] for auth in request.selected_authorities],
                        "authority_name": [auth["name"] for auth in request.selected_authorities],
                    }
                }
            )
            return IssueResponse(
                id=issue_id,
                message="Report submitted for quality review. Our team will verify the details shortly.",
                report={
                    "issue_id": issue_id,
                    "status": "needs_review",
                    "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
                    "report": report
                }
            )
        except Exception as e:
             logger.error(f"Failed to update issue status to needs_review: {e}")
             raise HTTPException(status_code=500, detail="Internal error updating issue status")
    
    report["responsible_authorities_or_parties"] = selected_authorities
    report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")
    
    recommended_actions = report.get("recommended_actions", [])
    if "recommended_actions" not in report:
        report["recommended_actions"] = recommended_actions
    
    email_success = False
    email_errors = []
    try:
        email_success = await send_authority_email(
            issue_id=issue_id,
            authorities=selected_authorities,
            issue_type=issue.get("issue_type", "Unknown Issue"),
            final_address=issue.get("address", "Unknown Address"),
            zip_code=issue.get("zip_code", "N/A"),
            timestamp_formatted=issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            report=report,
            confidence=issue.get("report", {}).get("issue_overview", {}).get("confidence", 0.0),
            category=issue.get("category", "Public"),
            timezone_name=issue.get("timezone_name", "UTC"),
            latitude=issue.get("latitude", 0.0),
            longitude=issue.get("longitude", 0.0),
            image_content=image_content,
            is_user_review=False
        )
        if not email_success:
            email_errors = [f"Email sending failed for {auth['email']}" for auth in selected_authorities]
            logger.warning(f"Email sending failed for issue {issue_id}: {email_errors}")
    except Exception as e:
        logger.error(f"Failed to send authority emails for issue {issue_id}: {str(e)}", exc_info=True)
        email_errors = [str(e)]
    
    try:
        await update_issue_status(issue_id, "completed")
        db = await get_db()
        await db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "report": report,
                    "authority_email": [auth["email"] for auth in selected_authorities],
                    "authority_name": [auth["name"] for auth in selected_authorities],
                    "email_status": "sent" if email_success else "failed",
                    "email_errors": email_errors,
                    "status": "submitted",
                    "decline_reason": None,
                    "decline_history": [],
                    "recommended_actions": recommended_actions
                }
            }
        )
        logger.debug(f"Issue {issue_id} updated with email_status: {'sent' if email_success else 'failed'}")
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id} status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update issue status: {str(e)}")
    
    logger.info(f"Issue {issue_id} submitted to authorities: {[auth['email'] for auth in selected_authorities]}. Email success: {email_success}")
    return IssueResponse(
        id=issue_id,
        message=f"Issue submitted successfully to selected authorities. {'Emails sent successfully' if email_success else 'Email sending failed: ' + '; '.join(email_errors)}",
        report={
            "issue_id": issue_id,
            "report": report,
            "authority_email": [auth["email"] for auth in selected_authorities],
            "authority_name": [auth["name"] for auth in selected_authorities],
            "recommended_actions": recommended_actions,
            "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "zip_code": issue.get("zip_code", "N/A"),
            "timezone_name": issue.get("timezone_name", "UTC")
        }
    )

@router.post("/issues/{issue_id}/accept", response_model=IssueResponse)
async def accept_issue(issue_id: str, request: AcceptRequest):
    logger.debug(f"Processing accept request for issue {issue_id}")
    try:
        db = await get_db()
        fs = await get_fs()
        logger.debug("Database and GridFS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    try:
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
            logger.error(f"Issue {issue_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        if issue.get("status") and issue.get("status") not in ["pending", "needs_review"]:
            logger.warning(f"Issue {issue_id} already processed with status {issue.get('status')}")
            raise HTTPException(status_code=400, detail="Issue already processed")
    except Exception as e:
        logger.error(f"Failed to fetch issue {issue_id} from database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch issue: {str(e)}")
    
    required_fields = ["issue_type", "address", "image_id", "report"]
    missing_fields = [field for field in required_fields if field not in issue or issue[field] is None]
    if missing_fields:
        logger.error(f"Issue {issue_id} missing required fields: {missing_fields}")
        raise HTTPException(status_code=400, detail=f"Issue missing required fields: {missing_fields}")
    
    report = request.edited_report if request.edited_report else issue["report"]
    if request.edited_report:
        try:
            EditedReport(**request.edited_report)
            report["template_fields"] = report.get("template_fields", issue["report"]["template_fields"])
            report["issue_overview"] = report.get("issue_overview", issue["report"]["issue_overview"])
            report["recommended_actions"] = report.get("recommended_actions", issue["report"]["recommended_actions"])
            report["detailed_analysis"] = report.get("detailed_analysis", issue["report"]["detailed_analysis"])
            report["responsible_authorities_or_parties"] = report.get("responsible_authorities_or_parties", issue["report"]["responsible_authorities_or_parties"])
            report["template_fields"].pop("tracking_link", None)
            report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")
        except Exception as e:
            logger.error(f"Invalid edited report for issue {issue_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid edited report: {str(e)}")
    else:
        report["template_fields"].pop("tracking_link", None)
        report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")

    # -------------------------------------------------------------------------
    # ACCEPT GUARD LOGIC (Combined Safety): 
    # Check for specific categories + Low Confidence -> Admin Review
    # -------------------------------------------------------------------------
    try:
        conf_val = 0.0
        
        # Extract confidence from the REPORT (which might be edited/different than issue root)
        conf_candidates = [
            report.get("template_fields", {}).get("confidence"),
            report.get("unified_report", {}).get("confidence"),
            report.get("issue_overview", {}).get("confidence"),
        ]
        
        valid_scores = []
        for c in conf_candidates:
            if c is None: continue
            try:
                s = str(c).strip().replace('%', '')
                v = float(s)
                if v <= 1.0: v = v * 100.0
                v = max(0.0, min(100.0, v))
                valid_scores.append(v)
            except: continue
        
        if valid_scores:
            conf_val = min(valid_scores)
        
        flagged_categories = [
            "bonfire", "controlled_fire", "festival", "ceremony", "burning_leaves", 
            "other", "unknown", "none"
        ]
        
        current_issue_type = report.get("issue_type", issue.get("issue_type", "unknown")).lower()
        if not current_issue_type or current_issue_type == "unknown":
             current_issue_type = report.get("issue_overview", {}).get("issue_type", "unknown").lower()

        is_flagged_category = current_issue_type in flagged_categories or "fire" in current_issue_type

        # If confidence < 70 OR flagged category -> Admin Review
        if conf_val < 70 or is_flagged_category:
             logger.info(f"ACCEPT GUARD: Issue {issue_id} flagged for review (Type={current_issue_type}, Conf={conf_val}%)")
             
             # Fail-Safe Update
             await update_issue_status(issue_id, "needs_review")
             
             # Return early success response indicating review
             return IssueResponse(
                id=issue_id,
                message="Report received. It has been flagged for internal quality assurance to verify details.",
                report={
                    "issue_id": issue_id,
                    "status": "needs_review",
                    "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in guard logic during accept: {e}")
        # FAIL-SAFE: Return review response on error
        return IssueResponse(
            id=issue_id,
            message="Report received. It has been flagged for internal quality assurance due to a system check.",
            report={
                "issue_id": issue_id,
                "status": "needs_review",
                "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            }
        )
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # FINAL GUARD LOGIC (Repeated for Safety): 
    # Check for specific categories + Low Confidence -> Admin Review
    # We must re-evaluate this here because user might have edited data or bypassed initial check.
    # -------------------------------------------------------------------------
    try:
        conf_val = 0.0
        
        # Extract confidence from the REPORT (which might be edited/different than issue root)
        conf_candidates = [
            report.get("template_fields", {}).get("confidence"),
            report.get("unified_report", {}).get("confidence"),
            report.get("issue_overview", {}).get("confidence"),
        ]
        
        valid_scores = []
        for c in conf_candidates:
            if c is None: continue
            try:
                s = str(c).strip().replace('%', '')
                v = float(s)
                if v <= 1.0: v = v * 100.0
                v = max(0.0, min(100.0, v))
                valid_scores.append(v)
            except: continue
        
        if valid_scores:
            conf_val = min(valid_scores)
        
        flagged_categories = [
            "bonfire", "controlled_fire", "festival", "ceremony", "burning_leaves", 
            "other", "unknown", "none"
        ]
        
        current_issue_type = report.get("issue_type", issue.get("issue_type", "unknown")).lower()
        # Also check nested issue type in overview if present, deeper check
        if not current_issue_type or current_issue_type == "unknown":
             current_issue_type = report.get("issue_overview", {}).get("issue_type", "unknown").lower()

        # Check if sensitive category OR 'fire' is in the type name (broad safety)
        is_flagged_category = current_issue_type in flagged_categories or "fire" in current_issue_type

        # If confidence < 70 OR flagged category -> Admin Review
        if conf_val < 70 or is_flagged_category:
             logger.info(f"FINAL SUBMIT GUARD: Issue {issue_id} flagged for review (Type={current_issue_type}, Conf={conf_val}%)")
             
             # Update status
             await update_issue_status(issue_id, "needs_review")
             
             # Return early success response indicating review
             return IssueResponse(
                id=issue_id,
                message="Report submitted for quality review. Our team will verify the details shortly.",
                report={
                    "issue_id": issue_id,
                    "status": "needs_review",
                    "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in guard logic during final submit: {e}")
        # FAIL-SAFE: If guard crashes but we suspected it might need review, 
        # or just to be safe, we should probably NOT send the email.
        # But if we can't determine, maybe it's safer to stop.
        # However, to avoid blocking legitimate submissions due to a bug, 
        # we usually pass. But since we had a critical bug (Invalid Status), 
        # let's assume if it reached here, we might want to default to 'needs_review' behavior 
        # if possible, or at least NOT automagically send email.
        
        # ACTUALLY, if we are in this block, 'conf_val' might reference before assignment error 
        # if it crashed early.
        # Let's return the review response to be safe (Fail Closed).
        return IssueResponse(
            id=issue_id,
            message="Report received. It has been flagged for internal quality assurance due to a system check.",
            report={
                "issue_id": issue_id,
                "status": "needs_review", # We claim this even if DB update failed (it might persist as pending)
                "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            }
        )
    # -------------------------------------------------------------------------
    
    recommended_actions = report.get("recommended_actions", [])
    if "recommended_actions" not in report:
        report["recommended_actions"] = recommended_actions
    
    try:
        # Get image content from GridFS
        gridout = await fs.open_download_stream(ObjectId(issue["image_id"]))
        image_content = await gridout.read()
        logger.debug(f"Image {issue['image_id']} retrieved for issue {issue_id}")
    except gridfs.errors.NoFile:
        logger.error(f"Image not found for image_id {issue['image_id']} in issue {issue_id}")
        raise HTTPException(status_code=404, detail=f"Image not found for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to fetch image for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(e)}")
    
    try:
        # Use selected authorities if provided, otherwise use recommended authorities
        if request.selected_authorities and len(request.selected_authorities) > 0:
            authorities = request.selected_authorities
            logger.info(f"Using selected authorities for issue {issue_id}: {[auth.get('name', 'Unknown') for auth in authorities]}")
        else:
            # Fallback to recommended authorities
            authorities = []
            if issue.get("zip_code"):
                authorities = get_authority_by_zip_code(issue["zip_code"], issue.get("issue_type", "Unknown Issue"), issue.get("category", "Public"))["responsible_authorities"]
            else:
                authorities = get_authority(
                    issue.get("address", "Unknown Address"),
                    issue.get("issue_type", "Unknown Issue"),
                    issue.get("latitude", 0.0),
                    issue.get("longitude", 0.0),
                    issue.get("category", "Public")
                )["responsible_authorities"]
            authorities = authorities or [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
            logger.info(f"Using recommended authorities for issue {issue_id}: {[auth.get('name', 'Unknown') for auth in authorities]}")
        
        logger.debug(f"Final authorities for issue {issue_id}: {[auth.get('email', 'No email') for auth in authorities]}")
    except Exception as e:
        logger.warning(f"Failed to fetch authorities for issue {issue_id}: {str(e)}. Using default authority.", exc_info=True)
        authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
    
    email_success = False
    email_errors = []
    try:
        email_success = await send_authority_email(
            issue_id=issue_id,
            authorities=authorities,
            issue_type=issue.get("issue_type", "Unknown Issue"),
            final_address=issue.get("address", "Unknown Address"),
            zip_code=issue.get("zip_code", "N/A"),
            timestamp_formatted=issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            report=report,
            confidence=issue.get("report", {}).get("issue_overview", {}).get("confidence", 0.0),
            category=issue.get("category", "Public"),
            timezone_name=issue.get("timezone_name", "UTC"),
            latitude=issue.get("latitude", 0.0),
            longitude=issue.get("longitude", 0.0),
            image_content=image_content,
            is_user_review=False
        )
        if not email_success:
            email_errors = [f"Email sending failed for {auth['email']}" for auth in authorities]
            logger.warning(f"Email sending failed for issue {issue_id}: {email_errors}")
    except Exception as e:
        logger.error(f"Failed to send authority emails for issue {issue_id}: {str(e)}", exc_info=True)
        email_errors = [str(e)]
    
    try:
        await update_issue_status(issue_id, "accepted")
        db = await get_db()
        await db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "report": report,
                    "authority_email": [auth["email"] for auth in authorities],
                    "authority_name": [auth["name"] for auth in authorities],
                    "email_status": "sent" if email_success else "failed",
                    "email_errors": email_errors,
                    "status": "accepted",
                    "decline_reason": None,
                    "decline_history": [],
                    "recommended_actions": recommended_actions
                }
            }
        )
        logger.debug(f"Issue {issue_id} updated with email_status: {'sent' if email_success else 'failed'}")
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id} status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update issue status: {str(e)}")
    
    logger.info(f"Issue {issue_id} accepted and reported to authorities: {[auth['email'] for auth in authorities]}. Email success: {email_success}")
    return IssueResponse(
        id=issue_id,
        message=f"Thank you for using eaiser! Issue accepted and {'emails sent successfully' if email_success else 'email sending failed: ' + '; '.join(email_errors)}",
        report={
            "issue_id": issue_id,
            "report": report,
            "authority_email": [auth["email"] for auth in authorities],
            "authority_name": [auth["name"] for auth in authorities],
            "recommended_actions": recommended_actions,
            "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "zip_code": issue.get("zip_code", "N/A"),
            "timezone_name": issue.get("timezone_name", "UTC")
        }
    )

@router.post("/issues/{issue_id}/decline", response_model=IssueResponse)
async def decline_issue(issue_id: str, request: DeclineRequest):
    logger.debug(f"Processing decline request for issue {issue_id} with reason: {request.decline_reason}")
    try:
        db = await get_db()
        fs = await get_fs()
        logger.debug("Database and GridFS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    try:
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
            logger.error(f"Issue {issue_id} not found in database")
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        if issue.get("status") != "pending":
            logger.warning(f"Issue {issue_id} already processed with status {issue.get('status')}")
            raise HTTPException(status_code=400, detail="Issue already processed")
    except Exception as e:
        logger.error(f"Failed to fetch issue {issue_id} from database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch issue: {str(e)}")
    
    required_fields = ["issue_type", "address", "image_id", "report"]
    missing_fields = [field for field in required_fields if field not in issue or issue[field] is None]
    if missing_fields:
        logger.error(f"Issue {issue_id} missing required fields: {missing_fields}")
        raise HTTPException(status_code=400, detail=f"Issue missing required fields: {missing_fields}")
    
    if not request.decline_reason or len(request.decline_reason.strip()) < 5:
        logger.error(f"Invalid decline reason for issue {issue_id}: {request.decline_reason}")
        raise HTTPException(status_code=400, detail="Decline reason must be at least 5 characters long")
    
    try:
        # Get image content from GridFS
        gridout = await fs.open_download_stream(ObjectId(issue["image_id"]))
        image_content = await gridout.read()
        logger.debug(f"Image {issue['image_id']} retrieved for issue {issue_id}")
    except gridfs.errors.NoFile:
        logger.error(f"Image not found for image_id {issue['image_id']} in issue {issue_id}")
        raise HTTPException(status_code=404, detail=f"Image not found for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to fetch image for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(e)}")
    
    report = request.edited_report if request.edited_report else issue["report"]
    if request.edited_report:
        try:
            EditedReport(**request.edited_report)
            report["template_fields"] = report.get("template_fields", issue["report"]["template_fields"])
            report["issue_overview"] = report.get("issue_overview", issue["report"]["issue_overview"])
            report["recommended_actions"] = report.get("recommended_actions", issue["report"]["recommended_actions"])
            report["detailed_analysis"] = report.get("detailed_analysis", issue["report"]["detailed_analysis"])
            report["responsible_authorities_or_parties"] = report.get("responsible_authorities_or_parties", issue["report"]["responsible_authorities_or_parties"])
            report["template_fields"].pop("tracking_link", None)
            report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")
        except Exception as e:
            logger.error(f"Invalid edited report for issue {issue_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid edited report: {str(e)}")
    else:
        report["template_fields"].pop("tracking_link", None)
        report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")
    
    try:
        updated_report = await generate_report(
            image_content=image_content,
            description="",
            issue_type=issue.get("issue_type", "Unknown Issue"),
            severity=issue.get("severity", "Medium"),
            address=issue.get("address", "Unknown Address"),
            zip_code=issue.get("zip_code", "N/A"),
            latitude=issue.get("latitude", 0.0),
            longitude=issue.get("longitude", 0.0),
            issue_id=issue_id,
            confidence=issue.get("report", {}).get("issue_overview", {}).get("confidence", 0.0),
            category=issue.get("category", "Public"),
            priority=issue.get("priority", "Medium"),
            decline_reason=request.decline_reason
        )
        updated_report["template_fields"].pop("tracking_link", None)
        updated_report["template_fields"]["zip_code"] = issue.get("zip_code", "N/A")
        
        recommended_actions = updated_report.get("recommended_actions", [])
        if "recommended_actions" not in updated_report:
            updated_report["recommended_actions"] = recommended_actions
        
        logger.debug(f"Updated report generated for issue {issue_id} with decline reason: {request.decline_reason}")
    except Exception as e:
        logger.error(f"Failed to generate updated report for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate updated report: {str(e)}")
    
    email_success = False
    email_errors = []
    try:
        user_authority = [{"name": "User", "email": issue.get("user_email", "eaiser@momntumai.com"), "type": "general"}]
        email_success = await send_authority_email(
            issue_id=issue_id,
            authorities=user_authority,
            issue_type=issue.get("issue_type", "Unknown Issue"),
            final_address=issue.get("address", "Unknown Address"),
            zip_code=issue.get("zip_code", "N/A"),
            timestamp_formatted=issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            report=updated_report,
            confidence=issue.get("report", {}).get("issue_overview", {}).get("confidence", 0.0),
            category=issue.get("category", "Public"),
            timezone_name=issue.get("timezone_name", "UTC"),
            latitude=issue.get("latitude", 0.0),
            longitude=issue.get("longitude", 0.0),
            image_content=image_content,
            decline_reason=request.decline_reason,
            is_user_review=True
        )
        if not email_success:
            email_errors = [f"Email sending failed for {user_authority[0]['email']}"]
            logger.warning(f"Email sending failed for issue {issue_id}: {email_errors}")
    except Exception as e:
        logger.error(f"Failed to send review email for issue {issue_id}: {str(e)}", exc_info=True)
        email_errors = [str(e)]
    
    try:
        decline_history = issue.get("decline_history", []) or []
        decline_history.append({
            "reason": request.decline_reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        db = await get_db()
        await db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "report": updated_report,
                    "decline_reason": request.decline_reason,
                    "decline_history": decline_history,
                    "email_status": "sent" if email_success else "failed",
                    "email_errors": email_errors,
                    "status": "pending",
                    "recommended_actions": recommended_actions
                }
            }
        )
        logger.debug(f"Issue {issue_id} updated with decline reason: {request.decline_reason}, email_status: {'sent' if email_success else 'failed'}")
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id} with decline reason: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update issue: {str(e)}")
    
    logger.info(f"Issue {issue_id} declined with reason: {request.decline_reason}. Updated report sent to user for review. Email success: {email_success}")
    return IssueResponse(
        id=issue_id,
        message=f"Issue declined with reason: {request.decline_reason}. Updated report sent for review. {'Emails sent successfully' if email_success else 'Email sending failed: ' + '; '.join(email_errors)}",
        report={
            "issue_id": issue_id,
            "report": updated_report,
            "authority_email": [issue.get("user_email", "eaiser@momntumai.com")],
            "authority_name": ["User"],
            "recommended_actions": recommended_actions,
            "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "zip_code": issue.get("zip_code", "N/A"),
            "timezone_name": issue.get("timezone_name", "UTC"),
            "decline_reason": request.decline_reason
        }
    )

@router.get("/issues", response_model=List[Issue])
async def list_issues(limit: int = 50, skip: int = 0):
    """
    List issues with pagination support for better performance.
    
    Args:
        limit: Maximum number of issues to return (default: 50, max: 100)
        skip: Number of issues to skip for pagination (default: 0)
    """
    # Validate pagination parameters
    if limit > 100:
        limit = 100  # Cap at 100 for performance
    if limit < 1:
        limit = 1
    if skip < 0:
        skip = 0
        
    try:
        # Use optimized get_issues function with pagination
        issues = await get_issues(limit=limit, skip=skip)
        formatted_issues = []
        
        for issue in issues:
            try:
                # Minimal timestamp processing
                timestamp = issue.get('timestamp')
                if isinstance(timestamp, datetime):
                    issue['timestamp'] = timestamp.isoformat()
                
                # Authority fields are already processed in get_issues()
                formatted_issues.append(Issue(**issue))
            except Exception as e:
                logger.warning(f"Skipping invalid issue {issue.get('_id')}: {str(e)}")
                continue
                
        logger.info(f"Retrieved {len(formatted_issues)} valid issues (limit: {limit}, skip: {skip})")
        return formatted_issues
    except Exception as e:
        logger.error(f"Failed to list issues: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list issues: {str(e)}")

@router.put("/issues/{issue_id}/status")
async def update_status(issue_id: str, status_update: IssueStatusUpdate):
    try:
        db = await get_db()
        updated = await update_issue_status(issue_id, status_update.status)
        if not updated:
            logger.error(f"Issue {issue_id} not found for status update")
            raise HTTPException(status_code=404, detail="Issue not found")
        logger.info(f"Status updated for issue {issue_id} to {status_update.status}")
        return {"message": "Status updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update status for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")

@router.get("/issues/{issue_id}/image")
async def get_issue_image(issue_id: str):
    try:
        db = await get_db()
        fs = await get_fs()
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
            logger.error(f"Issue {issue_id} not found for image retrieval")
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
            
        image_id = issue.get("image_id")
        if not image_id:
            logger.error(f"No image_id found for issue {issue_id}")
            raise HTTPException(status_code=404, detail=f"No image found for issue {issue_id}")
            
        try:
            gridout = await fs.open_download_stream(ObjectId(image_id))
            logger.debug(f"Retrieved image {image_id} for issue {issue_id}")
            return StreamingResponse(gridout, media_type="image/jpeg")
        except gridfs.errors.NoFile:
            logger.error(f"Image {image_id} not found in GridFS for issue {issue_id}")
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        except Exception as e:
            logger.error(f"Failed to retrieve image {image_id} for issue {issue_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to process image request for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image request: {str(e)}")

@router.post("/send-authority-emails")
async def send_authority_emails(request: EmailAuthoritiesRequest):
    """
    Send emails to multiple selected authorities for a specific issue
    """
    try:
        logger.info(f"🚨 AUTHORITY EMAIL ENDPOINT CALLED! 🚨")
        logger.info(f"🔥 DEBUG: Received request to send emails to {len(request.authorities)} authorities for issue {request.issue_id}")
        logger.info(f"🔥 DEBUG: Authorities received: {[str(auth.get('name', 'Unknown') if isinstance(auth, dict) else auth) + ' - ' + str(auth.get('email', 'No email') if isinstance(auth, dict) else 'No email') for auth in request.authorities]}")
        logger.info(f"🔥 DEBUG: Full authorities data: {request.authorities}")
        logger.info(f"🔥 DEBUG: Request zip code: {request.zip_code}")
        logger.info(f"🔥 DEBUG: Request issue ID: {request.issue_id}")
        
        # Get issue details from database (skip validation for testing)
        db = await get_db()
        issue = await db.issues.find_one({"_id": request.issue_id})
        if not issue:
            logger.warning(f"Issue {request.issue_id} not found in database, using mock data for testing")
            # Use mock issue data for testing
            issue = {
                "_id": request.issue_id,
                "issue_overview": {
                    "type": "Pothole",
                    "severity": "Medium",
                    "summary": "Test issue for email functionality"
                },
                "location": {
                    "address": "Test Address, Nashville, TN"
                },
                "zip_code": request.zip_code or "37013"
            }
        
        # Prepare email content
        report_data = request.report_data
        logger.info(f"🔥 DEBUG: report_data type: {type(report_data)}, content: {report_data}")
        
        # Handle different report_data structures - ensure it's a dict
        if isinstance(report_data, dict):
            logger.info("🔥 DEBUG: report_data is a dict, proceeding with dict operations")
            if 'issue_overview' in report_data:
                # Standard structure
                issue_overview = report_data.get('issue_overview', {})
                if isinstance(issue_overview, dict):
                    issue_type = issue_overview.get('type', 'Unknown')
                    severity = issue_overview.get('severity', 'Unknown')
                    summary = issue_overview.get('summary', 'No summary available')
                else:
                    issue_type = 'Unknown'
                    severity = 'Unknown'
                    summary = 'No summary available'
            else:
                # Direct structure from test data
                issue_type = report_data.get('category', 'Unknown')
                severity = report_data.get('severity', 'Unknown')
                summary = report_data.get('description', 'No summary available')
            
            location_data = report_data.get('location', 'Unknown location')
            if isinstance(location_data, dict):
                address = location_data.get('address', 'Unknown location')
            else:
                address = str(location_data)
        else:
            # Fallback if report_data is not a dict
            logger.warning(f"🔥 DEBUG: report_data is not a dict, type: {type(report_data)}")
            issue_type = 'Unknown'
            severity = 'Unknown'
            summary = 'No summary available'
            address = 'Unknown location'
        
        # Email subject and content
        subject = f"Public Issue Report - {issue_type} in {request.zip_code}"
        
        email_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px;">
                    Public Issue Report - {issue_type}
                </h2>
                
                <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #1e40af;">Issue Details</h3>
                    <p><strong>Type:</strong> {issue_type}</p>
                    <p><strong>Severity:</strong> {severity}</p>
                    <p><strong>Location:</strong> {address}</p>
                    <p><strong>Zip Code:</strong> {request.zip_code}</p>
                    <p><strong>Reported Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #1e40af;">Issue Summary</h3>
                    <p style="background: #fff; padding: 15px; border-left: 4px solid #2563eb; margin: 10px 0;">
                        {summary}
                    </p>
                </div>
                
                <div style="background: #ecfdf5; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #059669;">Action Required</h3>
                    <p>This issue has been reported by a community member and requires your attention. 
                    Please review the details and take appropriate action as per your department's protocols.</p>
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280;">
                    <p>This report was generated by Eaiser AI - Community Issue Reporting System</p>
                    <p>Report ID: {request.issue_id}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Send emails to all selected authorities
        successful_sends = 0
        failed_sends = 0
        send_results = []
        
        logger.info(f"🔥 DEBUG: Starting email loop for {len(request.authorities)} authorities")
        
        for i, authority in enumerate(request.authorities):
            try:
                authority_name = authority.get('name', 'Unknown Authority')
                authority_email = authority.get('email')
                authority_type = authority.get('type', 'Unknown Type')
                
                logger.info(f"🔥 DEBUG: Processing authority {i+1}/{len(request.authorities)}: {authority_name} - {authority_email}")
                
                if not authority_email:
                    logger.warning(f"🔥 DEBUG: No email found for authority: {authority_name}")
                    failed_sends += 1
                    send_results.append({
                        'authority': authority_name,
                        'status': 'failed',
                        'reason': 'No email address available'
                    })
                    continue
                
                # Personalize email for each authority
                personalized_subject = f"{subject} - Attention: {authority_name}"
                personalized_content = email_content.replace(
                    "This issue has been reported by a community member",
                    f"This issue has been reported by a community member and is being forwarded to {authority_name} ({authority_type})"
                )
                
                logger.info(f"🔥 DEBUG: Sending email to {authority_name} at {authority_email}")
                
                # Send email
                await send_email(
                    to_email=authority_email,
                    subject=personalized_subject,
                    html_content=personalized_content,
                    text_content=personalized_content.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n')
                )
                
                logger.info(f"🔥 DEBUG: Email sent successfully to {authority_name}")
                
                successful_sends += 1
                send_results.append({
                    'authority': authority_name,
                    'email': authority_email,
                    'status': 'sent',
                    'reason': 'Email sent successfully'
                })
                
                logger.info(f"Email sent successfully to {authority_name} ({authority_email})")
                
            except Exception as e:
                failed_sends += 1
                send_results.append({
                    'authority': authority.get('name', 'Unknown'),
                    'status': 'failed',
                    'reason': str(e)
                })
                logger.error(f"Failed to send email to {authority.get('name', 'Unknown')}: {str(e)}")
        
        # Update issue with email sending information
        try:
            await db.issues.update_one(
                {"_id": request.issue_id},
                {
                    "$set": {
                        "emails_sent": {
                            "timestamp": datetime.now(),
                            "authorities_contacted": len(request.authorities),
                            "successful_sends": successful_sends,
                            "failed_sends": failed_sends,
                            "send_results": send_results
                        }
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to update issue {request.issue_id} with email info: {str(e)}")
        
        logger.info(f"Email sending completed: {successful_sends} successful, {failed_sends} failed")
        
        return {
            "message": f"Emails processed for {len(request.authorities)} authorities",
            "successful_sends": successful_sends,
            "failed_sends": failed_sends,
            "send_results": send_results
        }
        
    except Exception as e:
        logger.error(f"Failed to send authority emails: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send emails: {str(e)}")

@router.get("/health")
async def health_check():
    try:
        db = await get_db()
        db.command("ping")
        logger.debug("Health check passed: database connected")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {str(e)}")