from fastapi import APIRouter, Request, HTTPException
from services.mongodb_service import get_db
from services.email_service import send_email
import logging
import re
import os
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

def extract_issue_id(text: str) -> str:
    """
    Extracts the Issue ID (Report ID) from the text.
    """
    if not text:
        return None
        
    match = re.search(r"eaiser-[\d-]+", text, re.IGNORECASE)
    if match:
        return match.group(0).lower().strip() 

    # 2. Try to find ID_ prefix
    match = re.search(r"ID_([a-zA-Z0-9]{5,20})", text, re.IGNORECASE)
    if match:
        return match.group(0).upper()
        
    # 3. Try to find ID: [VALUE] or (ID: [VALUE])
    match = re.search(r"ID[:\s]+([a-zA-Z0-9]{5,20})", text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        # Check if it was supposed to be ID_
        if not val.startswith("ID_") and len(val) <= 10:
             # If it's short, it might be the actual ID
             return val
        return val

    # 3. Last resort: any uppercase alphanumeric string of 5-10 chars that looks like an ID
    # This might be risky, so we only look in parentheses or brackets if possible
    match = re.search(r"[\(\[]ID[:\s]*([a-zA-Z0-9]+)[\)\]]", text, re.IGNORECASE)
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
        
        # 1. Extract Issue ID
        issue_id = extract_issue_id(subject) or extract_issue_id(body_text)
        
        if not issue_id:
            logger.warning(f"Could not extract Issue ID from email from {from_email}")
            return {"status": "ignored", "reason": "no_issue_id"}
            
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
        
        if is_from_authority:
            # Forward to User
            if not user_email:
                logger.warning(f"No user_email for issue {issue_id}, cannot forward authority reply.")
                return {"status": "ignored", "reason": "no_user_email"}
                
            forward_subject = f"RE: {subject}" if not subject.startswith("RE:") else subject
            forward_body = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; color: #333; border: 1px solid #e2e8f0; border-radius: 12px; background-color: #fffbeb;">
                <h2 style="color: #856404; margin-top: 0;">Message from Official Authority</h2>
                <p style="font-size: 16px; line-height: 1.6;">The authority managing your report <strong>#{issue_id[-6:]}</strong> has sent a response:</p>
                
                <div style="background-color: #ffffff; padding: 20px; border-left: 4px solid #fbbf24; border-radius: 8px; margin: 20px 0; font-style: italic; white-space: pre-wrap;">{body_text}</div>
                
                <p style="font-size: 14px; color: #666;">To reply back to the authority, simply <strong>reply to this email</strong>.</p>
                <hr style="border: 0; border-top: 1px solid #fde68a; margin: 20px 0;">
                <p style="font-size: 11px; color: #92400e; text-align: center;">Official EAiSER Automated Routing System</p>
            </div>
            """
            
            inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
            await send_email(user_email, forward_subject, forward_body, f"Authority reply for {issue_id}", reply_to=inbound_email)
            logger.info(f"✅ Forwarded authority reply for {issue_id} to user {user_email}")
            
        elif is_from_user:
            # Forward to Authorities
            if not authority_emails:
                logger.warning(f"No authority_emails for issue {issue_id}, cannot forward user reply.")
                return {"status": "ignored", "reason": "no_authority_emails"}
                
            forward_subject = f"RE: {subject}" if not subject.startswith("RE:") else subject
            forward_body = f"""
            <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; color: #333; border: 1px solid #e2e8f0; border-radius: 12px; background-color: #f8fafc;">
                <h2 style="color: #1e293b; margin-top: 0;">Citizen Follow-up: #{issue_id[-6:]}</h2>
                <p style="font-size: 16px; line-height: 1.6;">The reporter of incident <strong>#{issue_id}</strong> has sent a follow-up message:</p>
                
                <div style="background-color: #ffffff; padding: 20px; border-left: 4px solid #1e293b; border-radius: 8px; margin: 20px 0; font-style: italic; white-space: pre-wrap;">{body_text}</div>
                
                <p style="font-size: 14px; color: #666;">To coordinate with the citizen, simply <strong>reply to this email</strong>.</p>
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 20px 0;">
                <p style="font-size: 11px; color: #64748b; text-align: center;">Official EAiSER Automated Routing System</p>
            </div>
            """
            
            inbound_email = os.getenv("POSTMARK_INBOUND_EMAIL", "reports@inbound.eaiser.ai")
            for auth_email in authority_emails:
                await send_email(auth_email, forward_subject, forward_body, f"User follow-up for {issue_id}", reply_to=inbound_email)
                
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
