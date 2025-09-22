#!/usr/bin/env python3
"""
Simple SendGrid API test to verify API key permissions
"""
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment variables
load_dotenv()

def test_sendgrid_simple():
    """Test SendGrid API with a simple email"""
    api_key = os.getenv('SENDGRID_API_KEY')
    email_user = os.getenv('EMAIL_USER', 'raj@em7759.momntumai.com')
    
    if not api_key:
        print("❌ No SendGrid API key found")
        return False
    
    print("=== SendGrid Simple Test ===")
    print(f"Using API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"From Email: {email_user}")
    
    # Create a simple test email
    message = Mail(
        from_email=email_user,
        to_emails='test@example.com',  # This won't actually send
        subject='SendGrid Test Email',
        html_content='<strong>This is a test email from SendGrid</strong>'
    )
    
    try:
        # Initialize SendGrid client
        sg = SendGridAPIClient(api_key)
        
        # Try to send the email (this will fail with test email but should give us auth info)
        response = sg.send(message)
        
        print(f"✅ SendGrid API Response Status: {response.status_code}")
        print(f"Response Body: {response.body}")
        print(f"Response Headers: {response.headers}")
        
        return True
        
    except Exception as e:
        print(f"❌ SendGrid API Error: {str(e)}")
        print(f"Error Type: {type(e).__name__}")
        
        # Check if it's an authorization error
        if "401" in str(e) or "Unauthorized" in str(e):
            print("🔑 This is an authorization error - API key issue")
            print("💡 Solutions:")
            print("   1. Generate a new API key in SendGrid dashboard")
            print("   2. Ensure API key has 'Mail Send' permissions")
            print("   3. Check if API key is expired")
        
        return False

if __name__ == "__main__":
    test_sendgrid_simple()