import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_users():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.get_database("eaiser")
    
    users_col = db.get_collection("users")
    user = await users_col.find_one({"name": "chrishabh"})
    if not user:
        # try case insensitive or partial
        user = await users_col.find_one({"name": {"$regex": "chrishabh", "$options": "i"}})
        
    if user:
        print(f"User Found: {user.get('name')} | Email: {user.get('email')}")
        email = user.get('email')
        
        issues_col = db.get_collection("issues")
        # Check for user_email
        count1 = await issues_col.count_documents({"user_email": email})
        # Check for user_email as partial (if it was chrishabh)
        count2 = await issues_col.count_documents({"user_email": "chrishabh"})
        # Check for address/other fields if user_email is missing
        
        print(f"Issues with user_email={email}: {count1}")
        print(f"Issues with user_email='chrishabh': {count2}")
        
    else:
        print("User 'chrishabh' not found in users collection.")
        # List all users
        all_users = await users_col.find({}).to_list(length=10)
        for u in all_users:
            print(f"User: {u.get('name')} | Email: {u.get('email')}")

if __name__ == "__main__":
    asyncio.run(debug_users())
