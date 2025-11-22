import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional, Dict, Any
import base64
import asyncio
from fastapi import HTTPException
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, ContentId, Email
from python_http_client.exceptions import UnauthorizedError, BadRequestsError

# --------------------------------------------------------------------
# ✅ Environment & Logger Setup
# --------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger("email_service")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------
# ✅ Core Async Email Sender with Improved Error Handling
# --------------------------------------------------------------------
async def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None,
    retry: bool = True
) -> bool:
    """
    Sends an email via SendGrid API with inline images and attachments.
    Automatically retries once on transient failures.
    """
    email_user = os.getenv("EMAIL_USER", "no-reply@momntumai.com")
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    env = os.getenv("ENV", "development").lower()
    # Default dry_run to False so emails send by default unless explicitly disabled
    dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"

    # Only skip if dry_run is explicitly enabled
    if dry_run:
        logger.info(
            f"🧪 Email dry-run enabled. Would send to {to_email} subject='{subject}'."
        )
        return True

    # Validate environment variables for production sends
    if not all([email_user, sendgrid_api_key]):
        missing_vars = []
        if not email_user:
            missing_vars.append("EMAIL_USER")
        if not sendgrid_api_key:
            missing_vars.append("SENDGRID_API_KEY")

        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing email configuration: {', '.join(missing_vars)}")

    if not sendgrid_api_key.startswith('SG.'):
        logger.error("❌ Invalid SendGrid API key format. It must start with 'SG.'")
        return False

    # Build SendGrid mail object
    message = Mail(
        from_email=Email(email_user),
        to_emails=to_email,
        subject=subject,
        plain_text_content=text_content,
        html_content=html_content
    )

    # Add inline images
    if embedded_images:
        for cid, base64_data, mime_type in embedded_images:
            try:
                img_data = base64.b64decode(base64_data)
                encoded = base64.b64encode(img_data).decode()
                attachment = Attachment()
                attachment.file_content = FileContent(encoded)
                attachment.file_type = mime_type
                attachment.file_name = f"{cid}.{mime_type.split('/')[-1]}"
                attachment.disposition = "inline"
                attachment.content_id = ContentId(cid)
                message.add_attachment(attachment)
                logger.debug(f"🖼️ Embedded image {cid} added.")
            except Exception as e:
                logger.error(f"⚠️ Failed to embed image {cid}: {e}")

    # Add attachments
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                encoded = base64.b64encode(data).decode()
                attachment = Attachment()
                attachment.file_content = FileContent(encoded)
                attachment.file_type = "application/octet-stream"
                attachment.file_name = os.path.basename(file_path)
                attachment.disposition = "attachment"
                message.add_attachment(attachment)
                logger.debug(f"📎 Attached file: {file_path}")
            except Exception as e:
                logger.error(f"⚠️ Failed to attach file {file_path}: {e}")

    # Send via SendGrid
    try:
        logger.info(f"📤 Sending email FROM {email_user} TO {to_email} with subject: {subject}")
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)

        # Log response for diagnostics
        logger.info(f"📨 SendGrid response status: {response.status_code}")
        if hasattr(response, "body") and response.body:
            logger.debug(f"📄 SendGrid response body: {response.body.decode() if isinstance(response.body, bytes) else response.body}")

        if response.status_code in (200, 202):
            logger.info(f"✅ Email successfully sent to {to_email}")
            return True
        else:
            logger.warning(f"⚠️ SendGrid API returned {response.status_code} for {to_email}")
            if retry:
                logger.info("🔁 Retrying once after 2 seconds...")
                await asyncio.sleep(2)
                return await send_email(to_email, subject, html_content, text_content, attachments, embedded_images, retry=False)
            return False

    except UnauthorizedError:
        # Treat 401/403 as soft failures; log and do not crash
        logger.error("❌ SendGrid unauthorized — invalid API key.")
        return False
    except BadRequestsError as e:
        logger.error(f"❌ Bad Request to SendGrid: {e}")
        return False
    except Exception as e:
        err_text = str(e)
        if "403" in err_text or "Forbidden" in err_text:
            logger.error(f"🚫 SendGrid 403 Forbidden. The 'From' address ({email_user}) is likely not verified in SendGrid. Please verify it in SendGrid Settings > Sender Authentication.")
            return False
        logger.error(f"❌ Unexpected error sending email to {to_email}: {e}", exc_info=True)
        return False


# --------------------------------------------------------------------
# ✅ Synchronous Wrapper for Celery/Testing
# --------------------------------------------------------------------
def send_email_sync(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None
) -> bool:
    """Run async send_email synchronously (for local testing or Celery)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            send_email(to_email, subject, html_content, text_content, attachments, embedded_images)
        )
    except Exception as e:
        logger.error(f"❌ Sync wrapper failed: {e}")
        return False
    finally:
        loop.close()


# --------------------------------------------------------------------
# ✅ EmailService Class for Issue Notifications
# --------------------------------------------------------------------
class EmailService:
    async def send_issue_notification(self, authorities: List[Dict[str, Any]], issue_data: Dict[str, Any], issue_id: str) -> bool:
        """Send AI-detected issue notification emails to relevant authorities."""
        try:
            subject = f"New Issue Report #{issue_data.get('report_id', issue_id)}"
            text_content = (
                f"Issue ID: {issue_id}\n"
                f"Type: {issue_data.get('issue_type')}\n"
                f"Severity: {issue_data.get('severity')}\n"
                f"Description: {issue_data.get('description', 'N/A')}\n"
                f"Location: {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})\n"
            )
            html_content = (
                f"<h3>🛰️ New Issue Report Detected</h3>"
                f"<p><strong>Issue ID:</strong> {issue_id}</p>"
                f"<p><strong>Type:</strong> {issue_data.get('issue_type')}</p>"
                f"<p><strong>Severity:</strong> {issue_data.get('severity')}</p>"
                f"<p><strong>Description:</strong> {issue_data.get('description', 'N/A')}</p>"
                f"<p><strong>Location:</strong> {issue_data.get('address', 'N/A')} (ZIP: {issue_data.get('zip_code', 'N/A')})</p>"
            )

            sent_any = False
            for auth in authorities:
                to_email = auth.get('email') or auth.get('contact_email')
                if not to_email:
                    logger.warning(f"⚠️ Skipping authority without email: {auth}")
                    continue
                ok = await send_email(to_email, subject, html_content, text_content)
                sent_any = sent_any or ok

            logger.info(f"📬 Issue email notifications sent: {sent_any}")
            return sent_any

        except Exception as e:
            logger.error(f"❌ Issue notification failed: {e}", exc_info=True)
            return False


def get_email_service() -> EmailService:
    """Return a reusable instance of EmailService."""
    return EmailService()


# --------------------------------------------------------------------
# ✅ Send AI-Formatted Alert to Authorities (EAiSER Alert)
# --------------------------------------------------------------------
async def send_formatted_ai_alert(report: Dict[str, Any], background: bool = True) -> Dict[str, Any]:
    """
    Send the AI-generated formatted EAiSER alert to authorities.
    Uses the 'formatted_report' field from the AI report.
    """
    try:
        env = os.getenv("ENV", "development").lower()
        dry_run = os.getenv("EMAIL_DRY_RUN", "true").lower() == "true"
        formatted_content = report.get("formatted_report", "")
        issue_type = report.get("issue_overview", {}).get("type", "Issue")
        report_id = report.get("template_fields", {}).get("oid", "N/A")
        priority = report.get("template_fields", {}).get("priority", "N/A")
        subject = f"EAiSER Alert – {issue_type} (Priority: {priority}, ID: {report_id})"

        authorities = report.get("responsible_authorities_or_parties") or report.get("available_authorities") or []
        recipients = [a.get("email") for a in authorities if isinstance(a, dict) and a.get("email")]

        if not recipients:
            logger.warning("⚠️ No recipients found in AI report.")
            return {"status": "no_recipients", "recipients": []}

        # Only skip if dry_run is explicitly enabled
        if dry_run:
            logger.info(
                f"🧪 Email dry-run active. Skipping send to {len(recipients)} recipients"
            )
            return {"status": "dry_run", "recipients": recipients}

        async def _send(to_email: str):
            html = formatted_content.replace("\n", "<br>")
            return await send_email(to_email, subject, html, formatted_content)

        if background:
            for r in recipients:
                asyncio.create_task(_send(r))
            logger.info(f"📤 Dispatched EAiSER Alert to {len(recipients)} authorities (background mode)")
            return {"status": "dispatched", "recipients": recipients}
        else:
            sent, failed = [], []
            for r in recipients:
                ok = await _send(r)
                (sent if ok else failed).append(r)
            return {"status": "completed", "sent": sent, "failed": failed}

    except Exception as e:
        logger.error(f"❌ send_formatted_ai_alert failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}
