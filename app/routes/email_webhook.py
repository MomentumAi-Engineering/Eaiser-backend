from fastapi import APIRouter, Request, HTTPException
from services.mongodb_service import get_db
from services.email_service import send_email
import logging
import re
import os
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

def clean_email_body(text: str) -> str:
    """
    Removes common email reply headers and quoted text to show only the new message.
    """
    if not text:
        return ""
        
    # Common separators for reply history
    separators = [
        r"On\s+.*wrote:", 
        r"---+\s*Original Message\s*---+",
        r"From:\s+.*",
        r"Sent from my iPhone",
        r"Sent from my Android",
        r"Get Outlook for",
        r"---------- Forwarded message ----------"
    ]
    
    cleaned_text = text
    for sep in separators:
        # Split by separator and take the first part
        parts = re.split(sep, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        if parts:
            cleaned_text = parts[0]
            
    # Remove lines starting with > (standard email quotes)
    lines = cleaned_text.split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith('>')]
    
    return "\n".join(filtered_lines).strip()

def extract_issue_id(text: str) -> str:
    """
    Extracts the Issue ID (Report ID) from the text.
    Handles formats like: eaiser-1234, ID_ABC123, ID: ABC123, [ID: ABC123], #ABC123
    """
    if not text:
        return None
        
    # 1. Try eaiser- prefix
    match = re.search(r"eaiser-[\d-]+", text, re.IGNORECASE)
    if match:
        return match.group(0).lower().strip() 

    # 2. Try ID_ prefix
    match = re.search(r"ID_([a-zA-Z0-9]{5,20})", text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
        
    # 3. Try ID: [VALUE], ID [VALUE], or #VALUE
    # This covers "Report #A1B2C3D" or "ID: A1B2C3D"
    match = re.search(r"(?:ID[:\s]+|#)([a-zA-Z0-9]{5,20})", text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        return val

    # 4. Try bracketed/parenthesized formats like [ID: ABC123] or (ID ABC123)
    match = re.search(r"[\(\[](?:ID[:\s]*)?([a-zA-Z0-9]{5,20})[\)\]]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    return None

@router.post("/inbound")
async def handle_inbound_email(request: Request):
    """
    Webhook endpoint for Postmark Inbound Email.
    """
    try:
        data = await request.json()
        
        raw_from = data.get("From", "")
        # Extract email between brackets if present, else use raw string
        from_match = re.search(r'<(.*?)>', raw_from)
        from_email = from_match.group(1).lower() if from_match else raw_from.split()[-1].strip().lower()
        if not from_email:
             from_email = raw_from.lower()

        subject = data.get("Subject", "")
        body_text = data.get("TextBody", "")
        body_html = data.get("HtmlBody", "")
        
        logger.info(f"📥 Processing inbound email from {from_email}")
        logger.info(f"📝 Raw From: {raw_from} | Subject: {subject}")
        
        # Extract Attachments
        inbound_attachments = data.get("Attachments", [])
        raw_attachments_to_forward = []
        if inbound_attachments:
            logger.info(f"📎 Found {len(inbound_attachments)} attachments in inbound email.")
            # Prepare attachments for forwarding
            for att in inbound_attachments:
                name = att.get("Name")
                content = att.get("Content")
                ctype = att.get("ContentType")
                size = len(content) if content else 0
                logger.info(f"📎 Inbound attachment found: {name} ({ctype}) - Size: {size} chars")
                raw_attachments_to_forward.append({
                    "Name": name,
                    "Content": content,
                    "ContentType": ctype,
                    "ContentID": att.get("ContentID")
                })
        
        # 1. Extract Issue ID (Check Subject, then Text, then HTML)
        issue_id = extract_issue_id(subject) or extract_issue_id(body_text) or extract_issue_id(body_html)
        
        if not issue_id:
            logger.warning(f"⚠️ Could not extract Issue ID from email from {from_email}. Subject: {subject}")
            # Log first 100 chars of body for debugging
            logger.debug(f"Body snippet: {body_text[:100] if body_text else 'EMPTY'}")
            return {"status": "ignored", "reason": "no_issue_id"}
        
        logger.info(f"🔍 Extracted Issue ID: {issue_id}")
            
        # 2. Find Issue in DB
        db = await get_db()
        # Search in multiple possible ID fields including nested report fields
        issue = await db.issues.find_one({
            "$or": [
                {"_id": issue_id},
                {"report_id": issue_id},
                {"template_fields.oid": issue_id},
                {"report.template_fields.oid": issue_id},
                {"_id": {"$regex": f"^{re.escape(issue_id)}$", "$options": "i"}},
                {"report_id": {"$regex": f"^{re.escape(issue_id)}$", "$options": "i"}},
                {"report.template_fields.oid": {"$regex": f"^{re.escape(issue_id)}$", "$options": "i"}}
            ]
        })
        
        if not issue:
            logger.warning(f"❌ Issue {issue_id} not found in database for inbound email from {from_email}")
            return {"status": "ignored", "reason": f"issue_{issue_id}_not_found"}
            
        user_email = issue.get("user_email")
        authority_emails = issue.get("authority_email", [])
        
        # Normalize emails for comparison
        from_email_lower = from_email.lower()
        
        # 3. Determine Routing with better error handling for data types
        is_from_authority = False
        if authority_emails:
            for auth in authority_emails:
                auth_str = str(auth.get('email') if isinstance(auth, dict) else auth).lower()
                if auth_str in from_email_lower:
                    is_from_authority = True
                    break
        
        is_from_user = user_email and user_email.lower() in from_email_lower
        
        # Prepare display body (fallback to HTML if text is empty)
        display_body = body_text
        if not display_body or len(display_body.strip()) < 5:
            if body_html:
                # Simple strip tags for display
                display_body = re.sub(r'<[^>]+>', '', body_html)
            else:
                display_body = "[No text content]"
        
        # CLEAN THE BODY TO REMOVE QUOTES
        display_body = clean_email_body(display_body)

        if is_from_authority:
            # Forward to User (Citizen)
            if not user_email:
                logger.warning(f"No user_email for issue {issue_id}, cannot forward authority reply.")
                return {"status": "ignored", "reason": "no_user_email"}
                
            forward_subject = f"Official Response: Report #{issue_id}"
            
            # --- PREMIUM CITIZEN UPDATE TEMPLATE ---
            forward_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; line-height: 1.6; color: #1e293b; background-color: #f8fafc; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
        .wrapper {{ padding: 40px 20px; background-color: #f8fafc; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
        .header {{ background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); padding: 40px 30px; text-align: center; border-bottom: 1px solid #e2e8f0; position: relative; }}
        .header-accent {{ position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(to right, #fbbf24, #f59e0b); }}
        .header h1 {{ margin: 0; color: #0f172a; font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }}
        .id-badge {{ display: inline-block; padding: 6px 16px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 50px; font-size: 12px; font-weight: 700; color: #b45309; margin-top: 15px; letter-spacing: 0.5px; text-transform: uppercase; }}
        .content {{ padding: 40px 30px; }}
        .intro-text {{ font-size: 16px; color: #475569; margin-bottom: 25px; font-weight: 500; }}
        .message-box {{ background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6; padding: 25px; border-radius: 12px; font-size: 16px; color: #1e293b; line-height: 1.8; margin-bottom: 35px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .action-info {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; padding: 25px; text-align: center; color: #ffffff; }}
        .action-info p {{ margin: 0; font-size: 15px; font-weight: 500; color: #cbd5e1; }}
        .action-info strong {{ color: #ffffff; font-weight: 700; display: block; margin-top: 8px; font-size: 16px; }}
        .footer {{ padding: 30px; text-align: center; background: #f8fafc; border-top: 1px solid #e2e8f0; font-size: 12px; color: #94a3b8; }}
        .footer p {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="container">
            <div class="header">
                <div class="header-accent"></div>
                <h1>Official Authority Response</h1>
                <div class="id-badge">CIVIC ID: #{issue_id}</div>
            </div>
            <div class="content">
                <p class="intro-text">An official representative has reviewed your report and provided the following update:</p>
                <div class="message-box">
                    "{display_body}"
                </div>
                <div class="action-info">
                    <p>💬 Need to provide more details or ask a question?</p>
                    <strong>Simply reply directly to this email to continue the conversation.</strong>
                </div>
            </div>
            <div class="footer">
                <p>This is a secure automated transmission via the <strong>EAiSER Intelligence Engine</strong>.</p>
                <p>© 2026 EAiSER AI · Public Safety & Routing Protocol</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
            
            inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
            logger.info(f"📤 Forwarding authority reply to user {user_email} (Attachments: {len(raw_attachments_to_forward)})")
            await send_email(
                user_email, 
                forward_subject, 
                forward_body, 
                f"Authority reply for {issue_id}", 
                reply_to=inbound_email,
                raw_attachments=raw_attachments_to_forward
            )
            logger.info(f"✅ Forwarded authority reply for {issue_id} to user {user_email}")
            
        elif is_from_user:
            # Forward to Authorities
            if not authority_emails:
                logger.warning(f"No authority_emails for issue {issue_id}, cannot forward user reply.")
                return {"status": "ignored", "reason": "no_authority_emails"}
                
            forward_subject = f"NEW FOLLOW-UP: Incident #{issue_id}"
            
            # --- PREMIUM AUTHORITY ALERT TEMPLATE ---
            forward_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; line-height: 1.6; color: #1e293b; background-color: #0f172a; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }}
        .wrapper {{ padding: 40px 20px; background-color: #0f172a; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 24px; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); border: 1px solid #1e293b; }}
        .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 40px 30px; text-align: center; border-bottom: 4px solid #3b82f6; position: relative; }}
        .header h1 {{ margin: 0; color: #ffffff; font-size: 24px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }}
        .id-badge {{ display: inline-block; padding: 6px 16px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); border-radius: 50px; font-size: 12px; font-weight: 700; color: #60a5fa; margin-top: 15px; letter-spacing: 1px; text-transform: uppercase; }}
        .content {{ padding: 40px 30px; }}
        .section-tag {{ font-size: 11px; font-weight: 800; color: #3b82f6; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px; display: block; }}
        .message-box {{ background: #f8fafc; border-radius: 16px; padding: 30px; border: 1px solid #e2e8f0; font-size: 16px; color: #0f172a; line-height: 1.8; margin-bottom: 35px; }}
        .action-pane {{ background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 16px; padding: 25px; border: 1px solid #bfdbfe; }}
        .action-pane h4 {{ margin: 0 0 10px 0; color: #1e40af; font-size: 15px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }}
        .action-pane p {{ margin: 0; font-size: 15px; color: #1e3a8a; line-height: 1.6; font-weight: 500; }}
        .footer {{ padding: 30px; text-align: center; background: #f8fafc; border-top: 1px solid #e2e8f0; font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
        .footer p {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="container">
            <div class="header">
                <h1>Citizen Follow-up Received</h1>
                <div class="id-badge">Incident ID: {issue_id}</div>
            </div>
            <div class="content">
                <span class="section-tag">New Communication Logged</span>
                <div class="message-box">
                    {display_body}
                </div>
                <div class="action-pane">
                    <h4>💡 Operational Guidance</h4>
                    <p>Review the additional context above. You may <strong>reply directly to this email thread</strong> to securely coordinate with the citizen.</p>
                </div>
            </div>
            <div class="footer">
                <p>EAiSER Unified Routing Protocol v2.5</p>
                <p>Confidential • For Authorized Use Only</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
            
            inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
            logger.info(f"📤 Forwarding citizen follow-up for {issue_id} to {len(authority_emails)} authorities (Attachments: {len(raw_attachments_to_forward)})")
            for auth_email in authority_emails:
                await send_email(
                    auth_email, 
                    forward_subject, 
                    forward_body, 
                    f"User follow-up for {issue_id}", 
                    reply_to=inbound_email,
                    raw_attachments=raw_attachments_to_forward
                )
                
            logger.info(f"✅ Forwarded user follow-up for {issue_id} to {len(authority_emails)} authorities.")
            
        else:
            logger.warning(f"Email from {from_email} is neither authority nor user for issue {issue_id}")
            return {"status": "ignored", "reason": "unknown_sender"}
            
        # 4. Log Communication in Issue
        comm_entry = {
            "from": from_email,
            "role": "authority" if is_from_authority else "user",
            "timestamp": datetime.utcnow().isoformat(),
            "text": body_text[:2000] # Truncate for DB
        }
        await db.issues.update_one(
            {"_id": issue_id},
            {"$push": {"communication_log": comm_entry}}
        )
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error handling inbound email: {e}")
        return {"status": "error", "detail": str(e)}
