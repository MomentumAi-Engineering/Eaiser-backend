import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_db():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.get_database("eaiser") # or whatever the db name is
    
    # Try different DB names just in case
    db_names = await client.list_database_names()
    print(f"Databases: {db_names}")
    
    for db_name in ["eaiser", "snapfix"]:
        if db_name not in db_names: continue
        print(f"\nChecking DB: {db_name}")
        db = client.get_database(db_name)
        issues_col = db.get_collection("issues")
        
        count = await issues_col.count_documents({})
        print(f"Total issues: {count}")
        
        # Check all unique user_emails
        emails = await issues_col.distinct("user_email")
        print(f"Unique user_emails: {emails}")
        
        # Check for any issues where user_email might be stored differently
        sample = await issues_col.find_one({})
        if sample:
            print(f"Sample issue keys: {sample.keys()}")
            
        # Get count for a likely email
        for email in emails:
            if email:
                c = await issues_col.count_documents({"user_email": email})
                print(f"Issues for {email}: {c}")

if __name__ == "__main__":
    asyncio.run(debug_db())
