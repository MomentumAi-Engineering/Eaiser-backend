import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    print(f"Connecting to: {mongo_uri}")
    client = AsyncIOMotorClient(mongo_uri)
    
    db_names = await client.list_database_names()
    print(f"All Databases: {db_names}")
    
    target_dbs = [name for name in db_names if name not in ['admin', 'local', 'config']]
    
    for db_name in target_dbs:
        print(f"\n--- Checking DB: {db_name} ---")
        db = client.get_database(db_name)
        collections = await db.list_collection_names()
        print(f"Collections: {collections}")
        
        if "issues" in collections:
            issues_col = db.get_collection("issues")
            count = await issues_col.count_documents({})
            print(f"Total issues: {count}")
            
            emails = await issues_col.distinct("user_email")
            print(f"Distinct user_emails: {emails}")
            
            # Check for email field specifically
            user_emails_field = await issues_col.distinct("email")
            if user_emails_field:
                print(f"Distinct email fields: {user_emails_field}")

            # Sample document
            sample = await issues_col.find_one({})
            if sample:
                print(f"Sample document keys: {list(sample.keys())}")
                if 'user_email' in sample: print(f"Sample user_email: {sample['user_email']}")
                if 'email' in sample: print(f"Sample email: {sample['email']}")
                if 'status' in sample: print(f"Sample status: {sample['status']}")

if __name__ == "__main__":
    asyncio.run(debug_db())
