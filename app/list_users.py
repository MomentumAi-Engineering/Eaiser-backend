import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def list_users():
    uri = os.getenv('MONGO_URI')
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    
    # Check both potential databases
    for db_name in ["eaiser", "eaiser_db_user"]:
        db = client[db_name]
        colls = await db.list_collection_names()
        if "government_users" in colls:
            print(f"\n--- Checking DB: {db_name} ---")
            async for u in db["government_users"].find({}):
                print(f"USER: {u.get('name')} | ROLE: {u.get('role')} | EMAIL: {u.get('email')}")
                
            # FORCE FIX if found
            res = await db["government_users"].update_many(
                {"name": {"$regex": "^Rishav Kumar$", "$options": "i"}},
                {"$set": {"role": "super_admin"}}
            )
            if res.modified_count > 0:
                print(f"✅ Fixed {res.modified_count} users in {db_name}")

if __name__ == "__main__":
    asyncio.run(list_users())
