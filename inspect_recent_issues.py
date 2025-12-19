import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

from pathlib import Path

env_path = Path(__file__).parent / "app" / ".env"
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_NAME")

async def inspect_issues():
    print(f"Connecting to {MONGO_URI}...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    print("\n--- LAST 5 ISSUES (ANY STATUS) ---")
    cursor = db.issues.find({}).sort("timestamp", -1).limit(5)
    async for issue in cursor:
        print(f"\nID: {issue.get('_id')}")
        print(f"Status: {issue.get('status')}")
        print(f"Issue Type: {issue.get('issue_type')}")
        print(f"Dispatch Decision: {issue.get('dispatch_decision')}")
        
        report = issue.get("report", {})
        if isinstance(report, dict):
             conf = report.get("issue_overview", {}).get("confidence")
             print(f"Report Confidence: {conf} (Type: {type(conf)})")
             
             # Also check template_fields
             tmpl = report.get("template_fields", {}).get("confidence")
             print(f"Template Confidence: {tmpl} (Type: {type(tmpl)})")
             
        else:
             print("Report: <Not a dict>")

    print("\n--- ISSUES WITH status='needs_review' ---")
    count = await db.issues.count_documents({"status": "needs_review"})
    print(f"Count: {count}")
    
    print("\n--- ISSUES WITH status='submitted' AND LOW CONFIDENCE ---")
    # This mimics the fail-safe query I added
    cursor_scan = db.issues.find({
            "status": {"$in": ["submitted", "pending"]},
            "$or": [
                {"report.issue_overview.confidence": {"$lt": 70}},
                {"confidence": {"$lt": 70}}
            ]
        })
    found = await cursor_scan.to_list(length=5)
    print(f"Found via scan: {len(found)}")
    for f in found:
        print(f" - {f['_id']} ({f.get('status')})")

if __name__ == "__main__":
    asyncio.run(inspect_issues())
