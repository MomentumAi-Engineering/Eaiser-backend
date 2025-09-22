#!/usr/bin/env python3
"""
Test script to validate SendGrid API key configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test SendGrid API key configuration"""
    api_key = os.getenv('SENDGRID_API_KEY')
    
    print("=== SendGrid API Key Test ===")
    print(f"API Key exists: {bool(api_key)}")
    
    if api_key:
        print(f"API Key format valid: {api_key.startswith('SG.')}")
        print(f"API Key length: {len(api_key)}")
        print(f"API Key first 10 chars: {api_key[:10]}")
        print(f"API Key last 5 chars: ...{api_key[-5:]}")
    else:
        print("❌ No API key found in environment variables")
    
    # Check other email related env vars
    email_user = os.getenv('EMAIL_USER')
    print(f"\nEMAIL_USER: {email_user}")
    
    return bool(api_key) and api_key.startswith('SG.')

if __name__ == "__main__":
    is_valid = test_api_key()
    print(f"\n✅ API Key is valid: {is_valid}")