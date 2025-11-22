import requests
import json

# Check the raw issue data first
issue_id = "5b213f86-fcdf-47b3-89f7-486b3fc064f8"
url = f"http://localhost:10000/api/issues/{issue_id}"

# First, let's check all issues to see what's in the database
all_issues_url = "http://localhost:10000/api/issues"

try:
    # Check all issues first
    response = requests.get(all_issues_url)
    print(f"All issues status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} issues")
        if data:
            first_issue = data[0]
            print(f"First issue keys: {list(first_issue.keys())}")
            if 'processing_time_ms' in first_issue:
                print(f"processing_time_ms in first issue: {first_issue['processing_time_ms']}")
            else:
                print("processing_time_ms not in first issue")
    
    # Now check the specific issue
    print("\n" + "="*50)
    response = requests.get(url)
    print(f"Specific issue status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Issue keys: {list(data.keys())}")
        if 'processing_time_ms' in data:
            print(f"processing_time_ms: {data['processing_time_ms']}")
        if 'ai_report' in data:
            print(f"AI report found: {data['ai_report']}")
        else:
            print("No AI report found")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")