
import requests
import json

# Use the ID we found earlier: DDT62ZZ
# User Email is: chrishabh2002@gmail.com
# Authority Email is: chrishabh1000@gmail.com

url = "http://localhost:8005/api/email/inbound"
payload = {
    "From": "chrishabh1000@gmail.com",
    "Subject": "Re: Incident Report [ID: DDT62ZZ]",
    "TextBody": "Hello Citizen, this is a message from the authority regarding your report DDT62ZZ. We are investigating.",
    "HtmlBody": "<div>Hello Citizen, this is a message from the authority regarding your report DDT62ZZ. We are investigating.</div>"
}

print(f"🚀 Simulating Postmark Webhook to {url}...")
try:
    # We increase timeout to 30s to allow email sending
    response = requests.post(url, json=payload, timeout=30)
    print(f"✅ STATUS: {response.status_code}")
    print(f"✅ RESPONSE: {response.text}")
    print("\nNext: Check chrishabh2002@gmail.com for the forwarded message!")
except Exception as e:
    print(f"❌ ERROR: {e}")
