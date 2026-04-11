import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import motor.motor_asyncio

async def check():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True, tlsAllowInvalidCertificates=False)
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    db = client[db_name]
    
    # Check total issues
    count = await db["issues"].count_documents({})
    print(f"Total issues in DB: {count}")
    
    # Check sample emails
    cursor = db["issues"].find({}, {"user_email": 1, "_id": 0}).limit(10)
    emails = await cursor.to_list(10)
    print(f"Sample emails: {emails}")
    
    # Check for this specific user
    user_count = await db["issues"].count_documents({"user_email": "chrishabh2002@gmail.com"})
    print(f"Issues for chrishabh2002@gmail.com: {user_count}")
    
    # Check case-insensitive
    import re
    user_count_ci = await db["issues"].count_documents({"user_email": re.compile("chrishabh", re.IGNORECASE)})
    print(f"Issues matching 'chrishabh' (case-insensitive): {user_count_ci}")
    
    client.close()

asyncio.run(check())
