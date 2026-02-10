import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

async def run():
    uri = "mongodb+srv://eaiser_db_user:QXCk4SKf26lVr3Kv@eaiser-ai-v2.eyooh.mongodb.net/?retryWrites=true&w=majority&appName=EAISER-AI-V2"
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    db = client.eaiser_db_user
    
    email = "rishav@momntumai.com"
    # Find latest reports
    reports = await db.issues.find({"user_email": {"$regex": f"^{email}$", "$options": "i"}}).sort("timestamp", -1).limit(5).to_list(None)
    
    print(f"--- Latest 5 reports for {email} ---")
    for r in reports:
        print(f"ID: {r.get('_id')} | Status: {r.get('status')} | Conf: {r.get('confidence')} | Auth Routing: {r.get('automated_routing')} | Time: {r.get('timestamp')}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(run())
