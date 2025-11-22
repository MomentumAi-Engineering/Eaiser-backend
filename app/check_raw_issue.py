import requests
import json

# Check the raw issue data
issue_id = "67c5f4c0b0b8b6a3f4e8b0c2"
url = f"http://localhost:10000/api/issues/{issue_id}"

try:
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Issue keys: {list(data.keys())}")
        if 'processing_time_ms' in data:
            print(f"processing_time_ms already in data: {data['processing_time_ms']}")
        else:
            print("processing_time_ms not in raw data")
        print(f"Full response: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")