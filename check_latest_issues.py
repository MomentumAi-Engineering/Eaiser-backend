import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Load env
env_path = Path(__file__).parent / "app" / ".env"
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_NAME")

async def check_latest_issues():
    print(f"Connecting to DB...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db.issues
    
    print("\n=== LAST 10 ISSUES (BY TIMESTAMP) ===")
    cursor = collection.find({}).sort("timestamp", -1).limit(10)
    issues = await cursor.to_list(length=10)
    
    for i, issue in enumerate(issues, 1):
        ts = issue.get("timestamp")
        if isinstance(ts, datetime):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts)
            
        status = issue.get("status", "unknown")
        issue_type = issue.get("issue_type", "unknown")
        
        # Check confidence
        conf_val = None
        report = issue.get("report", {})
        if isinstance(report, dict):
            conf_val = report.get("issue_overview", {}).get("confidence")
            if conf_val is None:
                conf_val = report.get("template_fields", {}).get("confidence")
        
        print(f"\n{i}. ID: {str(issue['_id'])[-8:]}")
        print(f"   Timestamp: {ts_str}")
        print(f"   Status: {status}")
        print(f"   Type: {issue_type}")
        print(f"   Confidence: {conf_val}")
        print(f"   Dispatch Decision: {issue.get('dispatch_decision', 'N/A')}")
        
    print("\n=== ISSUES WITH needs_review STATUS ===")
    count = await collection.count_documents({"status": "needs_review"})
    print(f"Total: {count}")
    
    if count > 0:
        cursor = collection.find({"status": "needs_review"}).sort("timestamp", -1).limit(5)
        review_issues = await cursor.to_list(length=5)
        for issue in review_issues:
            ts = issue.get("timestamp")
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
            print(f"  - {str(issue['_id'])[-8:]} | {ts_str} | {issue.get('issue_type')}")

if __name__ == "__main__":
    asyncio.run(check_latest_issues())
