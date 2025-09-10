#!/usr/bin/env python3
"""
Real Email Test Script - Tests actual email sending to authorities
Yeh script real email send karega to check kya issue hai
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.email_service import send_email

load_dotenv()

async def test_real_email_send():
    """
    Test real email sending with actual SendGrid API call
    """
    print("🚀 Testing Real Email Send...")
    print("=" * 50)
    
    # Test email details
    test_email = "snapfix@momntumai.com"  # Safe test email
    subject = "🧪 Test Email from Eaiser Backend"
    html_content = """
    <html>
    <body>
        <h2>Test Email from Eaiser Backend</h2>
        <p>This is a test email to verify email functionality.</p>
        <p><strong>Issue Details:</strong></p>
        <ul>
            <li>Issue Type: Test Issue</li>
            <li>Location: Test Location</li>
            <li>Zip Code: 12345</li>
        </ul>
        <p>If you receive this email, the email service is working correctly!</p>
    </body>
    </html>
    """
    
    text_content = """
    Test Email from Eaiser Backend
    
    This is a test email to verify email functionality.
    
    Issue Details:
    - Issue Type: Test Issue
    - Location: Test Location
    - Zip Code: 12345
    
    If you receive this email, the email service is working correctly!
    """
    
    try:
        print(f"📧 Sending test email to: {test_email}")
        print(f"📝 Subject: {subject}")
        
        # Send the email
        result = await send_email(
            to_email=test_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )
        
        if result:
            print("✅ Email sent successfully!")
            print("🎉 Email service is working correctly")
        else:
            print("❌ Email sending failed")
            print("🔍 Check logs for detailed error information")
            
        return result
        
    except Exception as e:
        print(f"💥 Exception occurred: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("🧪 Real Email Test Script")
    print("=" * 50)
    
    # Check environment variables
    email_user = os.getenv("EMAIL_USER")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    
    print(f"📧 EMAIL_USER: {'✅ SET' if email_user else '❌ NOT SET'}")
    print(f"🔑 SENDGRID_API_KEY: {'✅ SET' if sendgrid_key else '❌ NOT SET'}")
    
    if sendgrid_key:
        print(f"🔍 API Key starts with: {sendgrid_key[:10]}...")
    
    print("=" * 50)
    
    # Run the test
    result = asyncio.run(test_real_email_send())
    
    print("=" * 50)
    if result:
        print("🎉 TEST PASSED: Email service is working!")
    else:
        print("❌ TEST FAILED: Email service has issues")
        print("🔧 Check SendGrid API key, account status, and logs")