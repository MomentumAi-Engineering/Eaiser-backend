
import asyncio
import os
import sys

sys.path.append(os.getcwd())
from services.mongodb_service import get_db

async def check_logs():
    db = await get_db()
    issue = await db.issues.find_one({"_id": "DDT62ZZ"})
    if issue:
        logs = issue.get("communication_log", [])
        print(f"LOG_COUNT:{len(logs)}")
        for log in logs:
            print(f"FROM:{log.get('from')} | TEXT:{log.get('text')}")
    else:
        print("ISSUE_NOT_FOUND")

if __name__ == "__main__":
    asyncio.run(check_logs())
