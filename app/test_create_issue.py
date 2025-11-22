#!/usr/bin/env python3
"""
Test script to create an issue with proper image upload
"""

import requests
import json
import base64
from PIL import Image
import io

def test_create_issue_with_image():
    """Test creating an issue with proper image upload"""
    
    # Read the image file
    image_path = "../test.png"
    
    # Prepare the form data
    files = {
        'image': ('test.png', open(image_path, 'rb'), 'image/png')
    }
    
    # Form data
    data = {
        'address': '123 Main St, Fairview, TN',
        'zip_code': '37062',
        'latitude': 35.0,
        'longitude': -87.0,
        'user_email': 'test@example.com',
        'category': 'public',
        'severity': 'high',
        'issue_type': 'fire'
    }
    
    try:
        # Make the request
        response = requests.post(
            'http://localhost:10000/api/issues',
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Issue created successfully!")
            print(f"Issue ID: {result.get('id')}")
            print(f"Message: {result.get('message')}")
            
            # Wait a bit for background processing
            import time
            time.sleep(10)
            
            # Check if the issue has an AI report
            issue_id = result.get('id')
            if issue_id:
                check_response = requests.get(f'http://localhost:10000/api/issues/{issue_id}')
                if check_response.status_code == 200:
                    issue_data = check_response.json()
                    print(f"Issue data: {json.dumps(issue_data, indent=2)}")
                    
                    # Check for AI report
                    if 'ai_report' in issue_data and issue_data['ai_report']:
                        print(f"🤖 AI Report: {issue_data['ai_report']}")
                    else:
                        print("❌ No AI report found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close the file
        files['image'][1].close()

if __name__ == "__main__":
    test_create_issue_with_image()