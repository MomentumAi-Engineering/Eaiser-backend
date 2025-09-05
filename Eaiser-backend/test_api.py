#!/usr/bin/env python3
"""
Test script for API endpoint performance testing
"""

import requests
import time
import json
from pathlib import Path

def test_issue_creation():
    """
    Test the /api/issues endpoint with proper multipart form data
    """
    url = "http://localhost:10000/api/issues"
    
    # Test image file
    image_path = Path("test_image.jpg")
    if not image_path.exists():
        print("❌ Test image not found!")
        return
    
    # Form data
    data = {
        'address': 'Test Address, New York',
        'latitude': '40.7128',
        'longitude': '-74.0060',
        'issue_type': 'pothole',
        'severity': 'medium',
        'category': 'public',
        'user_email': 'test@example.com'
    }
    
    # Files
    files = {
        'image': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    print("🚀 Testing /api/issues endpoint...")
    print(f"📍 URL: {url}")
    print(f"📊 Data: {data}")
    
    try:
        start_time = time.time()
        
        # Send POST request
        response = requests.post(url, data=data, files=files, timeout=30)
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"\n⏱️  Response Time: {response_time:.2f}ms")
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Issue created successfully")
            result = response.json()
            print(f"🆔 Issue ID: {result.get('id', 'N/A')}")
            print(f"💬 Message: {result.get('message', 'N/A')}")
            
            # Check if report was generated
            if 'report' in result:
                print("📋 Report generated successfully!")
                report = result['report']
                if 'processing_time_ms' in result:
                    print(f"⚡ Processing Time: {result['processing_time_ms']:.2f}ms")
            else:
                print("⚠️  No report in response")
                
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 30 seconds")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
    finally:
        # Close file
        if 'files' in locals():
            files['image'][1].close()

def test_get_report():
    """
    Test the /api/report GET endpoint
    """
    url = "http://localhost:10000/api/report"
    
    print("\n🚀 Testing /api/report endpoint...")
    print(f"📍 URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        
        print(f"⏱️  Response Time: {response_time:.2f}ms")
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Report endpoint working")
            result = response.json()
            print(f"📊 Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"💥 Error: {str(e)}")

if __name__ == "__main__":
    print("🧪 API Performance Test Suite")
    print("=" * 50)
    
    # Test issue creation (POST)
    test_issue_creation()
    
    # Test report endpoint (GET)
    test_get_report()
    
    print("\n✨ Testing completed!")
