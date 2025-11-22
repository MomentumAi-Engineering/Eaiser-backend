#!/usr/bin/env python3
"""
Test script for the /api/issues endpoint
"""

import asyncio
import aiohttp
import time

async def test_issues_endpoint():
    """Test the /api/issues endpoint"""
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://127.0.0.1:8000/api/issues') as response:
                end_time = time.time()
                print(f'Response Status: {response.status}')
                print(f'Response Time: {end_time - start_time:.2f} seconds')
                
                if response.status == 200:
                    data = await response.json()
                    print(f'✅ Request successful!')
                    print(f'Response type: {type(data)}')
                    print(f'Response preview: {str(data)[:300]}...')
                else:
                    text = await response.text()
                    print(f'❌ Request failed after {end_time - start_time:.2f} seconds')
                    print(f'Error Response: {text}')
                    
    except Exception as e:
        end_time = time.time()
        print(f'❌ Request failed after {end_time - start_time:.2f} seconds')
        print(f'Error: {e}')

if __name__ == "__main__":
    print("Testing /api/issues endpoint...")
    asyncio.run(test_issues_endpoint())