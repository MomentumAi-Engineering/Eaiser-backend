#!/usr/bin/env python3
"""
Test script to test the actual email sending endpoint
"""
import requests
import json

def test_email_endpoint():
    """Test the actual email sending endpoint"""
    
    # API endpoint
    url = "http://localhost:10000/api/send-authority-emails"
    
    # Test data - same format as frontend sends with all required fields
    test_data = {
        "issue_id": "test_issue_123",
        "authorities": [
            {
                "name": "Test Authority",
                "email": "snapfix@momntumai.com",  # Use valid test email
                "department": "Public Works"
            }
        ],
        "report_data": {
            "title": "Test Infrastructure Issue",
            "description": "This is a test issue for email functionality",
            "category": "Road Maintenance",
            "severity": "Medium",
            "location": "Test Location"
        },
        "zip_code": "12345",
        # Add the optional fields that were made optional in model
        "recommended_actions": [
            "Inspect the damaged area",
            "Schedule repair work",
            "Notify residents of timeline"
        ],
        "detailed_analysis": {
            "issue_type": "Infrastructure",
            "estimated_cost": "Medium",
            "urgency_level": "High"
        },
        "responsible_authorities_or_parties": [
            {
                "name": "Public Works Department",
                "contact": "test@example.com",
                "responsibility": "Road maintenance and repair"
            }
        ],
        "template_fields": {
            "greeting": "Dear Authority",
            "closing": "Best regards, SnapFix Team"
        }
    }
    
    print("=== Testing Email Endpoint ===")
    print(f"URL: {url}")
    print(f"Request Data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Send POST request
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n✅ Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Email endpoint working successfully!")
            print(f"Response: {response.json()}")
        elif response.status_code == 422:
            print("❌ 422 Validation Error - Check request format")
            print(f"Error Details: {response.text}")
        elif response.status_code == 500:
            print("❌ 500 Server Error - Check SendGrid configuration")
            print(f"Error Details: {response.text}")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Is the backend server running on port 10000?")
    except requests.exceptions.Timeout:
        print("❌ Request Timeout - Server took too long to respond")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_email_endpoint()