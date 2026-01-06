
import requests
import json

import random
import string
random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

payload = {
    "name": "Test User",
    "email": f"test_debug_{random_suffix}@example.com",
    "password": "password123"
}

try:
    print("Sending request to http://127.0.0.1:8000/api/auth/signup...")
    response = requests.post("http://127.0.0.1:8000/api/auth/signup", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
