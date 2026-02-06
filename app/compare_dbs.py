import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def compare_dbs():
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    
    for db_name in ["eaiser", "eaiser_db_user"]:
        db = client[db_name]
        try:
            issues = await db.issues.count_documents({})
            users = await db.users.count_documents({})
            print(f"DB: {db_name} | Issues: {issues} | Users: {users}")
            
            # Check user 'chrishabh' or similar
            user = await db.users.find_one({"name": {"$regex": "chrishabh", "$options": "i"}})
            if user:
                print(f"  -> User 'chrishabh' found in {db_name} as {user.get('email')}")
                # Check issues for this user
                issue_count = await db.issues.count_documents({"user_email": user.get('email')})
                print(f"  -> Issues for this user in {db_name}: {issue_count}")
        except Exception as e:
            print(f"Error checking {db_name}: {e}")

if __name__ == "__main__":
    asyncio.run(compare_dbs())
