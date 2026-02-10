import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import re

async def count_statuses():
    load_dotenv()
    uri = "mongodb+srv://eaiser_db_user:QXCk4SKf26lVr3Kv@eaiser-ai-v2.eyooh.mongodb.net/?retryWrites=true&w=majority&appName=EAISER-AI-V2"
    client = AsyncIOMotorClient(uri)
    db = client.eaiser_db_user
    
    user_email = "rishav@momntumai.com"
    filter_query = {"user_email": {"$regex": f"^{re.escape(user_email)}$", "$options": "i"}}
    
    # All statuses
    pipeline = [
        {"$match": filter_query},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    
    results = await db.issues.aggregate(pipeline).to_list(None)
    
    print(f"--- Status breakdown for {user_email} ---")
    total = 0
    for res in results:
        print(f"{res['_id']}: {res['count']}")
        total += res['count']
    print(f"Total entries in DB for this user: {total}")
    
    valid_pending = [
        "needs_review", "waiting_review", "pending_ai", "waiting_review_mobile", 
        "pending", "under_review", "under_admin_review", "submitted", "approved"
    ]
    valid_resolved = ["resolved", "completed", "analyzed", "rejected", "declined"]
    
    pending_count = sum(res['count'] for res in results if res['_id'] in valid_pending)
    resolved_count = sum(res['count'] for res in results if res['_id'] in valid_resolved)
    
    print(f"\nBreakdown based on NEW logic:")
    print(f"Total Reported (Pending + Resolved): {pending_count + resolved_count}")
    print(f"Pending: {pending_count}")
    print(f"Resolved (incl. Rejected/Declined): {resolved_count}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(count_statuses())
