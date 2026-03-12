
import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from services.mongodb_service import get_db

async def get_latest_issue():
    try:
        db = await get_db()
        issue = await db.issues.find_one(sort=[("timestamp", -1)])
        if issue:
            print(f"LATEST_ID:{issue['_id']}")
            print(f"USER_EMAIL:{issue.get('user_email')}")
            print(f"AUTH_EMAILS:{issue.get('authority_email')}")
        else:
            print("NO_ISSUES_FOUND")
    except Exception as e:
        print(f"ERROR:{e}")

if __name__ == "__main__":
    asyncio.run(get_latest_issue())
