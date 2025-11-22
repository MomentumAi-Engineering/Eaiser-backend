#!/usr/bin/env python3
"""
Check if AI report was generated for the issue
"""

import requests
import json

def check_issue():
    issue_id = "9e1ab950-58c6-4ae1-8b11-11147159e0a5"
    
    try:
        response = requests.get(f'http://localhost:10000/api/issues/{issue_id}')
        print(f'Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'AI Report: {data.get("ai_report", "Not found")}')
            print(f'Full response: {json.dumps(data, indent=2)}')
        else:
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    check_issue()