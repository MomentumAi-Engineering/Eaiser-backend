from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from datetime import datetime
try:
    from services.mongodb_service import get_db
    from services.email_service import send_email
except ImportError:
    from app.services.mongodb_service import get_db
    from app.services.email_service import send_email
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()

class CityInquiry(BaseModel):
    name: str
    city: str
    email: EmailStr
    role: str
    message: str

@router.post("/city-inquiry")
async def create_city_inquiry(inquiry: CityInquiry, background_tasks: BackgroundTasks):
    """
    Receives a 'For Cities' contact form submission.
    Saves it to the database and sends an email alert to the admin.
    """
    try:
        db = await get_db()
        inquiry_data = inquiry.dict()
        inquiry_data["created_at"] = datetime.utcnow()
        inquiry_data["status"] = "new"
        inquiry_data["type"] = "city_partner"

        # 1. Save to MongoDB
        await db["inquiries"].insert_one(inquiry_data)
        logger.info(f"📩 New city inquiry saved from {inquiry.email} regarding {inquiry.city}")

        # 2. Prepare Email for Admin
        admin_email = os.getenv("ADMIN_NOTIFICATION_EMAIL", "eaiser@momntumai.com")
        subject = f"🏙️ New City Partnership Request: {inquiry.city}"
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
            <h2 style="color: #fbbf24;">🏙️ New City Inquiry Received</h2>
            <p>You have a new partnership request from <strong>{inquiry.city}</strong>.</p>
            <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; color: #666;"><strong>Name:</strong></td>
                    <td style="padding: 8px 0;">{inquiry.name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;"><strong>Role:</strong></td>
                    <td style="padding: 8px 0;">{inquiry.role}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;"><strong>Email:</strong></td>
                    <td style="padding: 8px 0;"><a href="mailto:{inquiry.email}">{inquiry.email}</a></td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;"><strong>City:</strong></td>
                    <td style="padding: 8px 0;">{inquiry.city}</td>
                </tr>
            </table>
            <div style="margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 5px;">
                <strong>Message:</strong><br>
                <p style="white-space: pre-wrap;">{inquiry.message}</p>
            </div>
            <p style="font-size: 12px; color: #999; margin-top: 30px;">
                This inquiry has been logged in the database. Please reach out within 24 hours to maintain our commitment.
            </p>
        </div>
        """
        
        text_content = f"""
        New City Partnership Request
        Name: {inquiry.name}
        Role: {inquiry.role}
        City: {inquiry.city}
        Email: {inquiry.email}
        
        Message:
        {inquiry.message}
        """

        # 3. Send Email Alert (Background Task)
        background_tasks.add_task(send_email, admin_email, subject, html_content, text_content)

        return {"status": "success", "message": "Inquiry received. We will contact you shortly."}

    except Exception as e:
        logger.error(f"Error processing city inquiry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
