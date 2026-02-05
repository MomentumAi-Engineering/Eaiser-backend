
import sys
import os
import asyncio
from dotenv import load_dotenv

import logging
# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.DEBUG, 
                    handlers=[
                        logging.FileHandler("email_test_debug.log", mode='w', encoding='utf-8'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("email_service")
logger.setLevel(logging.DEBUG)

load_dotenv()

try:
    from app.services.email_service import send_email
except ImportError:
    # Try adding 'app' explicitly if needed, but running from backend root should be fine for 'app.services...'
    sys.path.append(os.path.join(os.getcwd(), 'app'))
    from services.email_service import send_email

async def main():
    print("--- Testing Email Service ---")
    
    api_key = os.getenv("SENDGRID_API_KEY")
    email_user = os.getenv("EMAIL_USER")
    
    print(f"EMAIL_USER: {email_user}")
    if api_key:
        # Mask key for privacy in logs
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        print(f"SENDGRID_API_KEY: {masked} (Length: {len(api_key)})")
    else:
        print("SENDGRID_API_KEY: Not Found!")
        
    dry_run = os.getenv("EMAIL_DRY_RUN", "false").lower() == "true"
    print(f"EMAIL_DRY_RUN: {dry_run}")

    recipient = "eaiser@momntumai.com"
    print(f"Attempting to send test email to {recipient}...")
    
    try:
        success = await send_email(
            to_email=recipient,
            subject="Test Email from EAiSER Console",
            html_content="<h1>Email System Test</h1><p>If you see this, the SendGrid integration is working.</p>",
            text_content="Email System Test. If you see this, the SendGrid integration is working."
        )
        
        if success:
            print("[SUCCESS] Email sent successfully!")
        else:
            print("[FAILURE] Email sending failed (Check logs/dry-run settings).")
    except Exception as e:
        print(f"[ERROR] Exception during sending: {e}")
        if hasattr(e, 'body'):
            print(f"[ERROR BODY] {e.body}")

if __name__ == "__main__":
    asyncio.run(main())
