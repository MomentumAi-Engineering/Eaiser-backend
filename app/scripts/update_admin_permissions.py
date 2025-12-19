import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

async def update_admin_permissions():
    print("🔧 Updating Admin Permissions...")
    
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser_db_user")
    
    client = AsyncIOMotorClient(uri)
    db = client[db_name]
    
    # Update all admins with missing fields
    result = await db.admins.update_many(
        {},
        {
            "$set": {
                "assigned_issues": [],
                "last_login": None
            }
        }
    )
    
    # Update super_admin permissions
    result = await db.admins.update_one(
        {"role": "super_admin"},
        {
            "$set": {
                "permissions": {
                    "can_approve": True,
                    "can_decline": True,
                    "can_assign": True,
                    "can_manage_team": True
                }
            }
        }
    )
    
    print(f"✅ Updated {result.modified_count} super admin(s)")
    
    # Update regular admins
    result = await db.admins.update_many(
        {"role": "admin"},
        {
            "$set": {
                "permissions": {
                    "can_approve": True,
                    "can_decline": True,
                    "can_assign": False,
                    "can_manage_team": False
                }
            }
        }
    )
    
    print(f"✅ Updated {result.modified_count} admin(s)")
    
    # Update viewers
    result = await db.admins.update_many(
        {"role": "viewer"},
        {
            "$set": {
                "permissions": {
                    "can_approve": False,
                    "can_decline": False,
                    "can_assign": False,
                    "can_manage_team": False
                }
            }
        }
    )
    
    print(f"✅ Updated {result.modified_count} viewer(s)")
    print("✨ All permissions updated!")

if __name__ == "__main__":
    asyncio.run(update_admin_permissions())
