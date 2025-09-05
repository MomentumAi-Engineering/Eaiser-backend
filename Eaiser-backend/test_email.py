#!/usr/bin/env python3
"""
🧪 Email Service Test Script
Tests SendGrid configuration and email functionality
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.email_service import send_email

# Load environment variables
load_dotenv()

async def test_email_service():
    """
    Test email service configuration and functionality
    """
    print("🧪 Testing Email Service Configuration...")
    print("=" * 50)
    
    # Check environment variables
    email_user = os.getenv("EMAIL_USER")
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    
    print(f"📧 EMAIL_USER: {'✅ SET' if email_user else '❌ NOT SET'}")
    print(f"🔑 SENDGRID_API_KEY: {'✅ SET' if sendgrid_api_key else '❌ NOT SET'}")
    
    if sendgrid_api_key:
        print(f"🔍 API Key Format: {'✅ VALID' if sendgrid_api_key.startswith('SG.') else '❌ INVALID'}")
        print(f"📏 API Key Length: {len(sendgrid_api_key)} characters")
    
    print("\n" + "=" * 50)
    
    if not all([email_user, sendgrid_api_key]):
        print("❌ Missing required environment variables!")
        print("\n📝 Required variables:")
        print("   - EMAIL_USER (sender email)")
        print("   - SENDGRID_API_KEY (SendGrid API key)")
        return False
    
    if not sendgrid_api_key.startswith('SG.'):
        print("❌ Invalid SendGrid API key format!")
        print("   API key should start with 'SG.'")
        return False
    
    # Test email sending (dry run)
    print("🚀 Testing email sending functionality...")
    
    try:
        # Test with a safe test email
        test_email = "test@example.com"  # This won't actually send
        test_subject = "🧪 Test Email from Eaiser AI"
        test_html = """
        <html>
        <body>
            <h2>🧪 Test Email</h2>
            <p>This is a test email from Eaiser AI backend.</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Status:</strong> Email service is working correctly!</p>
        </body>
        </html>
        """.format(timestamp=asyncio.get_event_loop().time())
        
        test_text = f"""
        🧪 Test Email
        
        This is a test email from Eaiser AI backend.
        Timestamp: {asyncio.get_event_loop().time()}
        Status: Email service is working correctly!
        """
        
        # Note: This will validate configuration but won't send to test@example.com
        result = await send_email(
            to_email=test_email,
            subject=test_subject,
            html_content=test_html,
            text_content=test_text
        )
        
        if result:
            print("✅ Email service configuration is valid!")
            print("📤 Email would be sent successfully in production")
        else:
            print("⚠️ Email service configuration has issues")
            print("🔍 Check the logs above for details")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing email service: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Email Service Test...")
    result = asyncio.run(test_email_service())
    
    print("\n" + "=" * 50)
    if result:
        print("✅ Email Service Test PASSED")
        print("🎉 SendGrid configuration is working correctly!")
    else:
        print("❌ Email Service Test FAILED")
        print("🔧 Please check your environment variables and API key")
    
    print("\n📋 Next Steps:")
    print("   1. Ensure SENDGRID_API_KEY is set in Render environment")
    print("   2. Verify API key is Production key (not Test key)")
    print("   3. Check SendGrid account status and billing")
    print("   4. Test with actual recipient email in development")