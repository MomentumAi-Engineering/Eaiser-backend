import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the parent directory to the path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.email_service import send_email
from app.utils.location import get_authority_by_zip_code

async def test_authority_email():
    """Test email sending to authorities"""
    load_dotenv()
    
    print("🔥 Testing authority email sending...")
    
    # Test zip code
    zip_code = "37013"
    issue_type = "pothole"
    category = "public"
    
    print(f"📍 Getting authorities for zip code: {zip_code}")
    
    # Get authorities using the same function as the backend
    authorities_result = get_authority_by_zip_code(zip_code, issue_type, category)
    
    print(f"🏛️ Authorities result: {authorities_result}")
    
    responsible_authorities = authorities_result.get('responsible_authorities', [])
    
    if not responsible_authorities:
        print("❌ No responsible authorities found!")
        return
    
    print(f"📧 Found {len(responsible_authorities)} responsible authorities")
    
    # Test email content
    subject = "Test Issue Report - Pothole in 37013"
    email_content = """
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Test Public Issue Report - Pothole</h2>
        <p><strong>Type:</strong> Pothole</p>
        <p><strong>Location:</strong> Test Location, 37013</p>
        <p><strong>Description:</strong> This is a test email to verify authority email sending.</p>
        <p>This is a test from Eaiser AI backend system.</p>
    </body>
    </html>
    """
    
    # Send email to each authority
    for i, authority in enumerate(responsible_authorities):
        authority_name = authority.get('name', 'Unknown Authority')
        authority_email = authority.get('email')
        authority_type = authority.get('type', 'Unknown Type')
        
        print(f"\n📤 Sending test email to authority {i+1}/{len(responsible_authorities)}:")
        print(f"   Name: {authority_name}")
        print(f"   Email: {authority_email}")
        print(f"   Type: {authority_type}")
        
        if not authority_email:
            print(f"   ❌ No email address for {authority_name}")
            continue
        
        try:
            # Send email
            await send_email(
                to_email=authority_email,
                subject=f"{subject} - Attention: {authority_name}",
                html_content=email_content,
                text_content="Test email for authority notification system."
            )
            print(f"   ✅ Email sent successfully to {authority_name}")
            
        except Exception as e:
            print(f"   ❌ Failed to send email to {authority_name}: {str(e)}")
    
    print("\n🎉 Authority email test completed!")

if __name__ == "__main__":
    asyncio.run(test_authority_email())