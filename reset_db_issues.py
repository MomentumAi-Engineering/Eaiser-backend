import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path

# Load env
env_path = Path(__file__).parent / 'app' / '.env'
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_NAME", "eaiser_db_user")

async def reset_database():
    print(f"🚀 Connecting to MongoDB: {DB_NAME}...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Collections to clear
    collections_to_clear = [
        "issues",
        "reports",
        "analytics",
        "fs.files",
        "fs.chunks",
        "notifications",
        "audit_logs"
    ]
    
    print(f"⚠️  WARNING: This will permanently delete all data in: {', '.join(collections_to_clear)}")
    confirm = input("Are you absolutely sure? (type 'YES' to continue): ")
    
    if confirm != "YES":
        print("❌ Reset cancelled.")
        return

    for coll_name in collections_to_clear:
        print(f"🧹 Clearing collection: {coll_name}...")
        await db[coll_name].delete_many({})
    
    print("\n✅ Database Reset Complete! Your issues and reports have been cleared.")
    print("✨ You can now start fresh with new reports.")

if __name__ == "__main__":
    asyncio.run(reset_database())
