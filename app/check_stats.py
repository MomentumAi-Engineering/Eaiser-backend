import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    results = await db.issues.aggregate(pipeline).to_list(None)
    for res in results:
        print(f"Status: {res['_id']}, Count: {res['count']}")

if __name__ == "__main__":
    asyncio.run(main())
