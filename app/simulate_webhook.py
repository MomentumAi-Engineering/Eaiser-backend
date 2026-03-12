
import requests
import json

url = "http://localhost:8005/api/email/inbound"
payload = {
    "From": "chrishabh1000@gmail.com",
    "Subject": "RE: EAiSER Alert – (ID: DDT62ZZ)",
    "TextBody": "This is a test reply from the simulated authority to the citizen.",
    "HtmlBody": "<div>This is a test reply from the simulated authority to the citizen.</div>"
}

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"STATUS_CODE:{response.status_code}")
    print(f"RESPONSE:{response.text}")
except Exception as e:
    print(f"ERROR:{e}")
