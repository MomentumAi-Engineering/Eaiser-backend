#!/usr/bin/env python3
"""
Check all issues to see if any have AI reports
"""

import requests
import json

def check_all_issues():
    try:
        response = requests.get('http://localhost:10000/api/issues')
        print(f'Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'Response type: {type(data)}')
            
            if isinstance(data, dict):
                print(f'Dict keys: {list(data.keys())}')
                if 'issues' in data:
                    issues = data['issues']
                    print(f'Found {len(issues)} issues')
                    if issues:
                        print('First issue keys:', list(issues[0].keys()))
                        print('First issue AI report:', issues[0].get('ai_report', 'Not found'))
                else:
                    print('Full response:', json.dumps(data, indent=2))
            elif isinstance(data, list):
                print(f'Found {len(data)} issues')
                if data:
                    print('First issue keys:', list(data[0].keys()))
                    print('First issue AI report:', data[0].get('ai_report', 'Not found'))
            else:
                print('Unexpected response format')
        else:
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    check_all_issues()