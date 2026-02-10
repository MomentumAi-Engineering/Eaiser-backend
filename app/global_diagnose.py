import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

async def run():
    # Use the URI we found in diagnose_stats.py
    uri = "mongodb+srv://eaiser_db_user:QXCk4SKf26lVr3Kv@eaiser-ai-v2.eyooh.mongodb.net/?retryWrites=true&w=majority&appName=EAISER-AI-V2"
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    db = client.eaiser_db_user
    
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    results = await db.issues.aggregate(pipeline).to_list(None)
    
    print("--- Global Status Breakdown ---")
    for res in results:
        print(f"{res['_id']}: {res['count']}")
    
    # Also check a few items to see if there are other status-like fields
    sample = await db.issues.find_one()
    print("\nSample keys:", sample.keys() if sample else "No data")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(run())
