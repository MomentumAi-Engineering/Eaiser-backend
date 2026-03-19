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
    
    # Try to find a user with issues
    sample_issue = await db.issues.find_one({"user_email": {"$ne": None}})
    if not sample_issue:
        print("No issues with non-null email found in DB.")
        return
        
    user_email = sample_issue["user_email"]
    print(f"Testing for user: {user_email}")
    
    emails_to_check = [user_email, user_email.lower(), user_email.upper()]
    filter_query = {
        "user_email": {"$in": emails_to_check},
        "$or": [
            {"is_submitted": True},
            {"status": {"$in": ["reported", "assigned", "in_progress", "working", "resolved", "needs_review", "submitted", "pending", "pending_review"]}}
        ]
    }
    
    count = await db.issues.count_documents(filter_query)
    print(f"Total issues found for user with optimized filter: {count}")

if __name__ == "__main__":
    asyncio.run(main())
