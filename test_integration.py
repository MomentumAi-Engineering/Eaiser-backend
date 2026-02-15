
import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to sys.path
app_path = Path(__file__).parent / "app"
sys.path.append(str(app_path))

from dotenv import load_dotenv
load_dotenv(dotenv_path=app_path / ".env", override=True)

from app.services.email_service import send_email

async def main():
    print("Testing Postmark integration via app/services/email_service.py...")
    success = await send_email(
        to_email="alert@momntumai.com",
        subject="Integration Test - Postmark",
        html_content="<h1>Service Test</h1><p>Testing import from email_service.py</p>",
        text_content="Service Test - Testing import from email_service.py"
    )
    if success:
        print("✅ SUCCESS: Email sent via Postmark service!")
    else:
        print("❌ FAILED: Could not send email via Postmark service. Check logs.")

if __name__ == "__main__":
    asyncio.run(main())
