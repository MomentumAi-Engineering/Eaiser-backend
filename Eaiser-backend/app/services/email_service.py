import os
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional
import base64
from fastapi import HTTPException
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    ContentId,
    Email,
)

load_dotenv()
logger = logging.getLogger(__name__)

def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    attachments: Optional[List[str]] = None,
    embedded_images: Optional[List[Tuple[str, str, str]]] = None
) -> bool:
    """
    Send an email with optional attachments and embedded images using SendGrid API.

    Args:
        to_email: Recipient email address.
        subject: Email subject.
        html_content: HTML content of the email, may include zip code.
        text_content: Plain text content of the email, may include zip code.
        attachments: List of file paths for attachments.
        embedded_images: List of tuples (cid, base64_data, mime_type) for embedded images (e.g., issue image, logo).

    Returns:
        bool: True if email was sent successfully, False otherwise.

    Raises:
        ValueError: If email configuration is missing in production.
        HTTPException: If email sending fails in production.
    """
    email_user = os.getenv("EMAIL_USER", "no-reply@momntumai.com")
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")

    if not all([email_user, sendgrid_api_key]):
        logger.error("Missing environment variables: EMAIL_USER, SENDGRID_API_KEY")
        if os.getenv("ENV") == "production":
            raise ValueError("Missing email configuration in production")
        logger.warning("Skipping email sending due to missing configuration")
        return False

    # Log presence of zip code in content for debugging
    has_zip_code = "Zip Code" in text_content or "Zip Code" in html_content
    logger.info(f"📧 Preparing email to {to_email} with subject '{subject}'. Zip code included: {has_zip_code}")

    # Create SendGrid Mail object
    message = Mail(
        from_email=Email(email_user),
        to_emails=to_email,
        subject=subject,
        plain_text_content=text_content,
        html_content=html_content
    )

    # Add embedded images
    if embedded_images:
        for cid, base64_data, mime_type in embedded_images:
            try:
                img_data = base64.b64decode(base64_data)
                encoded = base64.b64encode(img_data).decode()
                attachment = Attachment()
                attachment.file_content = FileContent(encoded)
                attachment.file_type = mime_type  # String, e.g., "image/png"
                attachment.file_name = f"{cid}.{mime_type.split('/')[-1]}"  # String, e.g., "logo.png"
                attachment.disposition = "inline"  # String
                attachment.content_id = ContentId(cid)
                message.add_attachment(attachment)
                logger.debug(f"Embedded image {cid} added to email")
            except Exception as e:
                logger.error(f"Failed to embed image {cid}: {str(e)}")
                if os.getenv("ENV") == "production":
                    raise HTTPException(status_code=500, detail=f"Failed to embed image {cid}: {str(e)}")
                continue

    # Add attachments
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                encoded = base64.b64encode(data).decode()
                attachment = Attachment()
                attachment.file_content = FileContent(encoded)
                attachment.file_type = "application/octet-stream"  # String
                attachment.file_name = os.path.basename(file_path)  # String
                attachment.disposition = "attachment"  # String
                message.add_attachment(attachment)
                logger.debug(f"Attached file {file_path} to email")
            except Exception as e:
                logger.error(f"Failed to attach file {file_path}: {str(e)}")
                if os.getenv("ENV") == "production":
                    raise HTTPException(status_code=500, detail=f"Failed to attach file {file_path}: {str(e)}")
                continue

    # Send email via SendGrid
    try:
        logger.info(f"📤 Sending email to {to_email} with subject: {subject} via SendGrid")
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        if response.status_code in (200, 202):
            logger.info(f"✅ Email sent successfully to {to_email} via SendGrid (Status: {response.status_code})")
            return True
        else:
            logger.error(f"❌ SendGrid API returned status {response.status_code} for {to_email}")
            if os.getenv("ENV") == "production":
                raise HTTPException(status_code=500, detail=f"SendGrid API error: Status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to send email to {to_email}: {str(e)}")
        if os.getenv("ENV") == "production":
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")
        return False