import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def debug_mongo():
    uri = os.getenv('MONGO_URI')
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    
    dbs = await client.list_database_names()
    print(f"Databases: {dbs}")
    
    for db_name in dbs:
        db = client[db_name]
        colls = await db.list_collection_names()
        if "government_users" in colls:
            print(f"--- DB: {db_name} (Has government_users) ---")
            async for user in db["government_users"].find({}):
                print(f"  User: {user.get('name')} | Role: {user.get('role')} | Email: {user.get('email')}")

if __name__ == "__main__":
    asyncio.run(debug_mongo())
