import asyncio
import motor.motor_asyncio
import os
from bson.objectid import ObjectId

async def run():
    uri = "mongodb+srv://eaiser_db_user:QXCk4SKf26lVr3Kv@eaiser-ai-v2.eyooh.mongodb.net/?retryWrites=true&w=majority&appName=EAISER-AI-V2"
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    db = client.eaiser_db_user
    
    email = "rishav@momntumai.com"
    report = await db.issues.find_one({"user_email": {"$regex": f"^{email}$", "$options": "i"}}, sort=[("timestamp", -1)])
    
    if report:
        print(f"Latest Report for {email}:")
        print(f"ID: {report.get('_id')}")
        print(f"Status: {report.get('status')}")
        print(f"Confidence: {report.get('confidence')}")
        print(f"Automated Routing: {report.get('automated_routing')}")
        print(f"AI Analysis: {bool(report.get('report'))}")
    else:
        print("No report found.")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(run())
