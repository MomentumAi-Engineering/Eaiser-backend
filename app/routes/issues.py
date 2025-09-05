from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from services.ai_service import classify_issue, generate_report
from services.email_service import send_email
from services.mongodb_service import store_issue, get_issues, update_issue_status, get_db, get_fs
from services.geocode_service import reverse_geocode, geocode_zip_code
from utils.location import get_authority, get_authority_by_zip_code
from utils.timezone import get_timezone_name
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

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
router = APIRouter()

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

class SubmitRequest(BaseModel):
    selected_authorities: List[Dict[str, str]]  # List of {name, email, type}

class EditedReport(BaseModel):
    issue_overview: Dict[str, Any]
    recommended_actions: List[str]
    detailed_analysis: Dict[str, Any]
    responsible_authorities_or_parties: List[Dict[str, Any]]
    template_fields: Dict[str, Any]

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
    
    severity_checkboxes = {
        "High": "□ High  ☑ Medium  □ Low" if report.get("issue_overview", {}).get("severity", "").lower() == "medium" else "☑ High  □ Medium  □ Low" if report.get("issue_overview", {}).get("severity", "").lower() == "high" else "□ High  □ Medium  ☑ Low",
        "Medium": "□ High  ☑ Medium  □ Low",
        "Low": "□ High  □ Medium  ☑ Low"
    }.get(report.get("issue_overview", {}).get("severity", "Medium").capitalize(), "□ High  ☑ Medium  □ Low")
    
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    
    if is_user_review:
        subject = f"Updated Report for {issue_type.title()} at {final_address} - Review Required"
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
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Please accept the report or provide further feedback by declining with a reason. Reply to this email or contact snapfix@momntumai.com.
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
• Recommended Action: Immediate inspection and fire suppression measures.
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact snapfix@momntumai.com for further details.
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
• Recommended Action: Deploy officers to investigate and secure the area.
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact snapfix@momntumai.com for further details.
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
• Recommended Action: Schedule maintenance and repair work.
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact snapfix@momntumai.com for further details.
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
• Recommended Action: Inspect and address issue promptly.
• Report ID: {report.get('template_fields', {}).get('oid', 'N/A')}
Photo Evidence:
• File: {report.get('template_fields', {}).get('image_filename', 'N/A')}
• AI Detection: "{report.get('template_fields', {}).get('ai_tag', 'N/A')}" - Confidence: {confidence}%
Contact snapfix@momntumai.com for further details.
Disclaimer: This AI-generated report may contain inaccuracies. Refer to the attached image for primary evidence.
"""
            }
        }
        template = templates.get(department_type, templates["general"])
        return template["subject"], template["text_content"]

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
        authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
    
    logo_base64 = get_logo_base64()
    issue_image_base64 = base64.b64encode(image_content).decode('utf-8')
    embedded_images = []
    if logo_base64:
        embedded_images.append(("momentumai_logo", logo_base64, "image/png"))
    embedded_images.append(("issue_image", issue_image_base64, "image/jpeg"))
    
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    
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
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        @keyframes fadeIn {{ 0% {{ opacity: 0; }} 100% {{ opacity: 1; }} }}
        @keyframes gradientBG {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
        @keyframes slideIn {{ 0% {{ transform: translateY(20px); opacity: 0; }} 100% {{ transform: translateY(0); opacity: 1; }} }}
        @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} 100% {{ transform: scale(1); }} }}
        
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            animation: fadeIn 1s ease-in; 
            background: linear-gradient(-45deg, #f8f9fa, #e9ecef, #dee2e6, #f8f9fa); 
            background-size: 400% 400%; 
            animation: gradientBG 15s ease infinite; 
            margin: 0; 
            padding: 20px; 
            color: #333; 
        }}
        
        .container {{ 
            max-width: 700px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); 
            overflow: hidden; 
            animation: slideIn 0.8s ease-out;
        }}
        
        .content {{ 
            padding: 30px; 
        }}
        
        .banner {{ 
            background: linear-gradient(90deg, #1a365d, #2a4365, #2c5282); 
            color: white; 
            padding: 20px; 
            text-align: center; 
            font-weight: bold; 
            font-size: 18px;
            position: relative;
            overflow: hidden;
        }}
        
        .banner::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.1) 75%, transparent 75%, transparent);
            background-size: 30px 30px;
            animation: slide 20s linear infinite;
        }}
        
        @keyframes slide {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 30px 30px; }}
        }}
        
        .header {{ 
            text-align: center; 
            padding: 20px 0; 
            position: relative;
        }}
        
        .logo {{ 
            height: 60px; 
            animation: pulse 2s infinite; 
        }}
        
        .section {{ 
            margin-bottom: 25px; 
            border-bottom: 1px solid #eee; 
            padding-bottom: 20px; 
            animation: slideIn 0.8s ease-out;
            animation-fill-mode: both;
        }}
        
        .section:nth-child(1) {{ animation-delay: 0.1s; }}
        .section:nth-child(2) {{ animation-delay: 0.2s; }}
        .section:nth-child(3) {{ animation-delay: 0.3s; }}
        .section:nth-child(4) {{ animation-delay: 0.4s; }}
        .section:nth-child(5) {{ animation-delay: 0.5s; }}
        .section:nth-child(6) {{ animation-delay: 0.6s; }}
        
        .section-title {{ 
            color: #1a365d; 
            font-weight: 600; 
            margin-bottom: 15px; 
            display: flex; 
            align-items: center;
            font-size: 18px;
        }}
        
        .priority-box {{ 
            display: inline-block; 
            padding: 5px 12px; 
            border-radius: 20px; 
            font-weight: bold; 
            margin-left: 10px; 
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .high-priority {{ 
            background-color: #e53e3e; 
            color: white; 
        }}
        
        .medium-priority {{ 
            background-color: #dd6b20; 
            color: white; 
        }}
        
        .low-priority {{ 
            background-color: #38a169; 
            color: white; 
        }}
        
        .emoji {{ 
            font-size: 22px; 
            margin-right: 10px; 
            filter: drop-shadow(0 2px 2px rgba(0,0,0,0.1));
        }}
        
        .footer {{ 
            text-align: center; 
            font-size: 12px; 
            color: #777; 
            padding: 20px; 
            background: #f7fafc; 
        }}
        
        .ai-tag {{ 
            background: #ebf8ff; 
            padding: 10px 15px; 
            border-radius: 8px; 
            display: inline-block; 
            margin: 10px 0; 
            font-style: italic; 
            border-left: 4px solid #3182ce;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .confidence-meter {{ 
            height: 12px; 
            background: #e2e8f0; 
            border-radius: 6px; 
            margin-top: 8px; 
            overflow: hidden; 
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .confidence-level {{ 
            height: 100%; 
            background: linear-gradient(90deg, #3182ce, #63b3ed); 
            width: {confidence}%; 
            border-radius: 6px;
            box-shadow: 0 0 8px rgba(49, 130, 206, 0.5);
        }}
        
        .issue-image {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .issue-image:hover {{
            transform: scale(1.02);
        }}
        
        .actions-container {{
            background: linear-gradient(to right, #f7fafc, #edf2f7);
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid #3182ce;
        }}
        
        .action-item {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 15px;
            padding: 12px 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .action-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .action-item:last-child {{
            margin-bottom: 0;
        }}
        
        .action-icon {{
            background: #3182ce;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            flex-shrink: 0;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(49, 130, 206, 0.3);
        }}
        
        .action-text {{
            font-size: 15px;
            line-height: 1.5;
        }}
        
        .action-urgency {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
            color: white;
        }}
        
        .urgency-immediate {{
            background-color: #e53e3e;
        }}
        
        .urgency-high {{
            background-color: #dd6b20;
        }}
        
        .urgency-medium {{
            background-color: #3182ce;
        }}
        
        .urgency-low {{
            background-color: #38a169;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        tr {{
            border-bottom: 1px solid #eee;
        }}
        
        td {{
            padding: 12px 0;
        }}
        
        td:first-child {{
            font-weight: 600;
            color: #4a5568;
        }}
        
        td:last-child {{
            text-align: right;
        }}
        
        .cta-box {{
            background: linear-gradient(135deg, #ebf8ff, #bee3f8);
            padding: 20px;
            border-radius: 10px;
            margin-top: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 4px solid #3182ce;
        }}
        
        .cta-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c5282;
            font-size: 16px;
        }}
        
        .btn {{
            display: inline-block;
            background: #3182ce;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            margin-top: 10px;
            font-weight: 500;
            transition: background 0.2s ease;
        }}
        
        .btn:hover {{
            background: #2c5282;
        }}
        
        .btn-secondary {{
            background: #718096;
        }}
        
        .btn-secondary:hover {{
            background: #4a5568;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="banner">
            🚨 {'Updated Report for Review' if is_user_review else 'New Infrastructure Issue Detected'} 🚨
        </div>
        <div class="content">
            <div class="header">
                {'<img src="cid:momentumai_logo" alt="MomentumAI Logo" class="logo">' if logo_base64 else '<h2 style="color: #1a365d;">MomentumAI</h2>'}
                <h2 style="color: #1a365d; margin-top: 10px;">SnapFix AI Report</h2>
            </div>
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">👋</span> {'Hello User' if is_user_review else 'Hello Team'}
                </div>
                <p>{'Please review the updated report for a' if is_user_review else 'Our AI has detected a'} <strong>{issue_type.title()}</strong> issue{' that requires your attention' if not is_user_review else ''}.</p>
                {'<p><strong>Decline Reason:</strong> ' + decline_reason + '</p>' if decline_reason and is_user_review else ''}
            </div>
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">📜</span> Executive Summary
                </div>
                <p>{report.get('issue_overview', {}).get('summary_explanation', 'No summary available')}</p>
            </div>
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">📍</span> Location Details
                </div>
                <p><strong>Address:</strong> {final_address}</p>
                <p><strong>Zip Code:</strong> {zip_code}</p>
                <p><strong>Coordinates:</strong> {latitude if latitude else 'N/A'}, {longitude if longitude else 'N/A'}</p>
                <p><strong>Map Link:</strong> <a href="{map_link}" target="_blank" style="color: #3182ce; text-decoration: none;">{map_link if map_link.startswith('http') else 'No coordinates provided'}</a></p>
            </div>
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">📋</span> Report Summary
                </div>
                <table>
                    <tr>
                        <td>Report ID</td>
                        <td>{report.get('template_fields', {}).get('oid', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Issue Type</td>
                        <td>{category}</td>
                    </tr>
                    <tr>
                        <td>Time Reported</td>
                        <td>{timestamp_formatted} {timezone_name}</td>
                    </tr>
                    <tr>
                        <td>Priority</td>
                        <td>
                            <span class="priority-box {'high-priority' if report.get('template_fields', {}).get('priority', 'Medium') == 'High' else 'medium-priority' if report.get('template_fields', {}).get('priority', 'Medium') == 'Medium' else 'low-priority'}">
                                {report.get('template_fields', {}).get('priority', 'Medium')} Priority
                            </span>
                        </td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">🔍</span> AI Analysis
                </div>
                <div class="ai-tag">
                    "{report.get('template_fields', {}).get('ai_tag', 'N/A')}"
                </div>
                <p><strong>Confidence Level:</strong> {confidence}%</p>
                <div class="confidence-meter">
                    <div class="confidence-level"></div>
                </div>
                <p><strong>Potential Impact:</strong> {report.get('detailed_analysis', {}).get('potential_consequences_if_ignored', 'N/A')}</p>
                <p><strong>Urgency Reason:</strong> {report.get('detailed_analysis', {}).get('public_safety_risk', 'Unknown').title()} risk to public safety</p>
                <p><strong>Location Context:</strong> {final_address}</p>
                {'<p><strong>Feedback:</strong> ' + feedback_str + '</p>' if is_user_review else ''}
            </div>
            
            {"""
            <div class="section">
                <div class="section-title">
                    <span class="emoji">🛠️</span> Recommended Actions
                </div>
                <div class="actions-container">
                    """ + recommended_actions_html + """
                </div>
            </div>
            """ if not is_user_review else ''}
            
            <div class="section">
                <div class="section-title">
                    <span class="emoji">📸</span> Photo Evidence
                </div>
                <img src="cid:issue_image" alt="Issue Image" class="issue-image">
                <p><small>File: {report.get('template_fields', {}).get('image_filename', 'N/A')}</small></p>
            </div>
            
            <div class="cta-box">
                <div class="cta-title">📩 {'Action Required' if is_user_review else 'Need to respond?'}</div>
                <p>{'Please review the updated report and either accept it or provide further feedback by declining with a reason.' if is_user_review else 'Please take appropriate action and contact us if needed.'} Reply to this email or forward to <a href="mailto:snapfix@momntumai.com" style="color: #3182ce;">snapfix@momntumai.com</a> with your comments.</p>
                <a href="{map_link}" class="btn btn-secondary">View on Map</a>
            </div>
        </div>
        
        <div class="footer">
            <p>This report was submitted via SnapFix AI by MomntumAI</p>
            <p>© {report.get('template_fields', {}).get('timestamp', datetime.utcnow().strftime('%Y-%m-%d')).split('-')[0]} MomntumAI | All Rights Reserved</p>
            <p style="font-size: 10px; color: #aaa;">This is an automated message. Please do not reply directly to this email.</p>
        </div>
    </div>
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
            success = send_email(
                to_email=authority.get("email", "snapfix@momntumai.com"),
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                attachments=None,
                embedded_images=embedded_images
            )
            if success:
                successful_emails.append(authority.get("email", "snapfix@momntumai.com"))
                logger.info(f"Email sent successfully to [redacted] for {authority.get('type', 'general')}")
            else:
                logger.warning(f"Email sending failed for [redacted] without raising an exception")
                errors.append(f"Email sending failed for {authority.get('email', 'snapfix@momntumai.com')}")
        except Exception as e:
            logger.error(f"Failed to send email to [redacted]: {str(e)}", exc_info=True)
            errors.append(f"Failed to send email to {authority.get('email', 'snapfix@momntumai.com')}: {str(e)}")
    
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
    image: UploadFile = File(...),
    address: str = Form(''),
    zip_code: Optional[str] = Form(None),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    user_email: Optional[str] = Form(None),
    category: str = Form('public'),
    severity: str = Form('medium'),
    issue_type: str = Form('other')
):
    logger.debug(f"Creating issue with address: {address}, zip: {zip_code}, lat: {latitude}, lon: {longitude}, user_email: [redacted]")
    try:
        db = await get_db()
        fs = await get_fs()
        logger.debug("Database and GridFS initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")
    
    if not image.content_type.startswith("image/"):
        logger.error(f"Invalid image format: {image.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        image_content = await image.read()
        logger.debug(f"Image read successfully, size: {len(image_content)} bytes")
    except Exception as e:
        logger.error(f"Failed to read image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")
    
    try:
        issue_type, severity, confidence, category, priority = await classify_issue(image_content, "")
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
            description="",
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
        report["template_fields"]["zip_code"] = zip_code or "N/A"
        report["template_fields"]["address"] = final_address or "Not specified"
        
        recommended_actions = report.get("recommended_actions", [])
        if "recommended_actions" not in report:
            report["recommended_actions"] = recommended_actions
        
        logger.debug(f"Report generated for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to generate report for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    try:
        authority_data = get_authority_by_zip_code(zip_code, issue_type, category) if zip_code else get_authority(final_address, issue_type, latitude, longitude, category)
        responsible_authorities = authority_data.get("responsible_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])
        available_authorities = authority_data.get("available_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])
        
        responsible_authorities = [
            {**{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}, **auth}
            for auth in responsible_authorities
        ]
        available_authorities = [
            {**{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}, **auth}
            for auth in available_authorities
        ]
        
        authority_emails = [auth["email"] for auth in responsible_authorities]
        authority_names = [auth["name"] for auth in responsible_authorities]
        logger.debug(f"Responsible authorities fetched: {authority_emails}")
        logger.debug(f"Available authorities fetched: {[auth['email'] for auth in available_authorities]}")
    except Exception as e:
        logger.warning(f"Failed to fetch authorities: {str(e)}. Using default authority.", exc_info=True)
        responsible_authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
        available_authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
        authority_emails = ["snapfix@momntumai.com"]
        authority_names = ["City Department"]
    
    timezone_name = get_timezone_name(latitude, longitude) or "UTC"
    timestamp = datetime.utcnow().isoformat()
    timestamp_formatted = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        final_zip_code = zip_code if zip_code else "N/A"
        final_address = final_address if final_address else "Unknown Address"
        final_latitude = latitude if latitude else 0.0
        final_longitude = longitude if longitude else 0.0
        image_id = await store_issue(
            db=db,
            fs=fs,
            issue_id=issue_id,
            image_content=image_content,
            report=report,
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
    
    try:
        user_authority = [{"name": "User", "email": user_email or "snapfix@momntumai.com", "type": "general"}]
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
        await update_issue_status(issue_id, "submitted")
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
        authorities = authorities or [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
        logger.debug(f"Authorities for issue {issue_id}: {[auth['email'] for auth in authorities]}")
    except Exception as e:
        logger.warning(f"Failed to fetch authorities for issue {issue_id}: {str(e)}. Using default authority.", exc_info=True)
        authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
    
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
        message=f"Thank you for using SnapFix! Issue accepted and {'emails sent successfully' if email_success else 'email sending failed: ' + '; '.join(email_errors)}",
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
        user_authority = [{"name": "User", "email": issue.get("user_email", "snapfix@momntumai.com"), "type": "general"}]
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
            "authority_email": [issue.get("user_email", "snapfix@momntumai.com")],
            "authority_name": ["User"],
            "recommended_actions": recommended_actions,
            "timestamp_formatted": issue.get("timestamp_formatted", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "zip_code": issue.get("zip_code", "N/A"),
            "timezone_name": issue.get("timezone_name", "UTC"),
            "decline_reason": request.decline_reason
        }
    )

@router.get("/issues", response_model=List[Issue])
async def list_issues():
    try:
        db = await get_db()
        issues = await get_issues()
        formatted_issues = []
        for issue in issues:
            try:
                timestamp = issue.get('timestamp')
                if isinstance(timestamp, datetime):
                    issue['timestamp'] = timestamp.isoformat()
                
                authority_email = issue.get("authority_email", ["snapfix@momntumai.com"])
                if isinstance(authority_email, list):
                    authority_email = [str(email) for email in authority_email if email is not None and isinstance(email, str)]
                    if not authority_email:
                        authority_email = ["snapfix@momntumai.com"]
                else:
                    authority_email = [str(authority_email)] if authority_email else ["snapfix@momntumai.com"]
                    
                authority_name = issue.get("authority_name", ["City Department"])
                if isinstance(authority_name, list):
                    authority_name = [str(name) for name in authority_name if name is not None and isinstance(name, str)]
                    if not authority_name:
                        authority_name = ["City Department"]
                else:
                    authority_name = [str(authority_name)] if authority_name else ["City Department"]
                    
                issue["authority_email"] = authority_email
                issue["authority_name"] = authority_name
                
                formatted_issues.append(Issue(**issue))
            except Exception as e:
                logger.warning(f"Skipping invalid issue {issue.get('_id')}: {str(e)}", exc_info=True)
                continue
                
        logger.info(f"Retrieved {len(formatted_issues)} valid issues")
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