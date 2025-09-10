#!/usr/bin/env python3
"""
Test script to verify authorities endpoint is working
"""

import requests
import json

def test_authorities_endpoint():
    """Test the authorities endpoint with a known zip code"""
    
    base_url = "http://localhost:10000"
    zip_code = "37013"  # Test zip code
    
    print(f"🔍 Testing authorities endpoint for zip code: {zip_code}")
    
    try:
        # Make API call to authorities endpoint
        url = f"{base_url}/api/authorities/{zip_code}"
        print(f"📡 Making request to: {url}")
        
        response = requests.get(url, timeout=10)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Authorities data received:")
            print(json.dumps(data, indent=2))
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("🔧 Make sure backend is running on port 10000")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting authorities endpoint test...")
    success = test_authorities_endpoint()
    
    if success:
        print("\n✅ Authorities endpoint is working correctly!")
    else:
        print("\n❌ Authorities endpoint test failed!")