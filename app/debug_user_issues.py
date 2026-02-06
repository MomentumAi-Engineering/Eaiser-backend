import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_user_issues():
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.get_database("eaiser_db_user")
    
    email = "rishav@momntumai.com"
    issues_col = db.get_collection("issues")
    
    issues = await issues_col.find({"user_email": email}).to_list(length=100)
    print(f"Total issues for {email}: {len(issues)}")
    
    status_counts = {}
    for i in issues:
        s = i.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        
    print(f"Status distribution: {status_counts}")
    
    if issues:
        print(f"Sample issue 1: {issues[0].get('_id')} | Status: {issues[0].get('status')} | Type: {issues[0].get('issue_type')}")

if __name__ == "__main__":
    asyncio.run(debug_user_issues())
