import requests
import json

BASE_URL = "http://127.0.0.1:10000/api/admin/review"

def test_get_pending():
    print("\n🔍 Testing GET /pending...")
    try:
        response = requests.get(f"{BASE_URL}/pending")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Pending reviews count: {len(data)}")
            # print(data)
            return data
        else:
            print(f"Error: {response.text}")
            return []
    except Exception as e:
        print(f"Request failed: {e}")
        return []

def test_approve(issue_id):
    if not issue_id:
        print("\n⚠️ Skipping Approve test (no issue_id)")
        return
        
    print(f"\n✅ Testing POST /approve for {issue_id}...")
    payload = {
        "issue_id": issue_id,
        "admin_id": "test_admin_01",
        "notes": "Approved via automated test"
    }
    try:
        response = requests.post(f"{BASE_URL}/approve", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

def test_deactivate(email):
    print(f"\n🚫 Testing POST /deactivate-user for {email}...")
    payload = {
        "user_email": email,
        "reason": "Test deactivation",
        "admin_id": "test_admin_01"
    }
    try:
        response = requests.post(f"{BASE_URL}/deactivate-user", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    pending = test_get_pending()
    
    if pending:
        # Test approve on the first one
        target_id = pending[0].get("_id")
        test_approve(target_id)
    else:
        print("No pending issues to test approval.")
    
    test_deactivate("spammer@example.com")
