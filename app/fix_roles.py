import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def fix_role():
    # Use the same logic as the backend to find the DB name
    mongo_uri = os.getenv('MONGO_URI') or os.getenv('MONGODB_URL') or 'mongodb://localhost:27017'
    db_name = os.getenv('MONGODB_NAME', 'eaiser')
    
    print(f"Connecting to: {mongo_uri}")
    print(f"Target Database: {db_name}")
    
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    # Check if user exists first
    user = await db["government_users"].find_one({"name": "Rishav Kumar"})
    if user:
        print(f"Found user: {user.get('name')} with current role: {user.get('role')}")
        res = await db["government_users"].update_one(
            {"_id": user["_id"]},
            {"$set": {"role": "super_admin"}}
        )
        print(f"Updated Rishav Kumar to super_admin. Modified: {res.modified_count}")
    else:
        print("User 'Rishav Kumar' not found. Checking for case-insensitive match...")
        user = await db["government_users"].find_one({"name": {"$regex": "^Rishav Kumar$", "$options": "i"}})
        if user:
             print(f"Found case-insensitive user: {user.get('name')} with role: {user.get('role')}")
             res = await db["government_users"].update_one(
                {"_id": user["_id"]},
                {"$set": {"role": "super_admin"}}
             )
             print(f"Updated user to super_admin. Modified: {res.modified_count}")
        else:
             print("Rishav Kumar still not found. Listing all users to be sure:")
             async for u in db["government_users"].find({}):
                 print(f"- {u.get('name')} ({u.get('role')})")

if __name__ == "__main__":
    asyncio.run(fix_role())
