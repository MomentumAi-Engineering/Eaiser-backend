
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_mongo():
    uri = os.getenv("MONGO_URI") or "mongodb://localhost:27017/eaiser"
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    print(f"Connecting to {uri}, DB: {db_name}")
    
    client = AsyncIOMotorClient(uri)
    db = client[db_name]
    
    # Check for needs_review
    count_review = await db.issues.count_documents({"status": "needs_review"})
    print(f"Count of 'needs_review': {count_review}")
    
    cursor = db.issues.find({"status": "needs_review"})
    async for doc in cursor:
        print(f" - Found Review Issue: {doc.get('_id')} | Type: {doc.get('issue_type')} | Conf: {doc.get('report', {}).get('issue_overview', {}).get('confidence', 'N/A')}")

    # Check for submitted but low confidence
    print("\nChecking for 'submitted' issues with potential low confidence (manual scan)...")
    count_submitted = await db.issues.count_documents({"status": "submitted"})
    print(f"Total 'submitted': {count_submitted}")
    
    cursor = db.issues.find({"status": "submitted"}).sort("timestamp", -1).limit(20)
    async for doc in cursor:
        conf = doc.get('report', {}).get('issue_overview', {}).get('confidence')
        print(f" - Submitted Issue: {doc.get('_id')} | Status: {doc.get('status')} | Conf: {conf}")

if __name__ == "__main__":
    asyncio.run(debug_mongo())
