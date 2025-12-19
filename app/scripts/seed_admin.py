import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from utils.security import get_password_hash
from dotenv import load_dotenv
from datetime import datetime
import os

# Load environment variables
load_dotenv()

async def seed_admin():
    print("🔧 Seeding Super Admin...")
    print("=" * 50)
    
    try:
        # Direct connection with env variables
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_NAME", "eaiser_db_user")
        
        print(f"📊 Connecting to database: {db_name}")
        client = AsyncIOMotorClient(uri)
        db = client[db_name]
        
        # Test connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    email = "admin@eaiser.ai"
    password = "admin123"  # Initial default password
    
    # Check if admin exists
    existing = await db.admins.find_one({"email": email})
    if existing:
        print(f"⚠️  Super Admin {email} already exists.")
        print(f"   Role: {existing.get('role', 'N/A')}")
        print(f"   Permissions: {existing.get('permissions', {})}")
        return

    print(f"👤 Creating Super Admin: {email}")
    admin_data = {
        "email": email,
        "password_hash": get_password_hash(password),
        "name": "Super Admin",
        "role": "super_admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "assigned_issues": [],
        "permissions": {
            "can_approve": True,
            "can_decline": True,
            "can_assign": True,
            "can_manage_team": True
        },
        "last_login": None
    }
    
    await db.admins.insert_one(admin_data)
    print("=" * 50)
    print("✅ Super Admin created successfully!")
    print(f"📧 Email: {email}")
    print(f"🔑 Password: {password}")
    print(f"🎯 Role: super_admin")
    print(f"✨ Permissions: Full Access")
    print("=" * 50)
    print("⚠️  Please change the password after first login!")

if __name__ == "__main__":
    asyncio.run(seed_admin())
