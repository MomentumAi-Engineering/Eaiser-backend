
import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path.cwd() / "app"))

async def test_env():
    print("--- Testing Environment ---")
    try:
        import passlib
        print("passlib: OK")
    except ImportError as e:
        print(f"passlib: MISSING ({e})")

    try:
        import bcrypt
        print("bcrypt: OK")
    except ImportError as e:
        print(f"bcrypt: MISSING ({e})")

    try:
        from services.mongodb_service import init_db, get_db
        print("Imports from mongodb_service: OK")
        
        # Mock ENV if need be, but likely they are in .env file
        # Check if .env is loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL")
        print(f"MONGO_URI from env: {uri}")
        
        print("Attempting DB Init...")
        success = await init_db()
        print(f"DB Init Result: {success}")
        
        if success:
            db = await get_db()
            print(f"DB Object: {db}")
            try:
                await db.command("ping")
                print("DB Ping: SUCCESS")
            except Exception as e:
                print(f"DB Ping: FAILED ({e})")
        else:
            print("DB Init returned False")

    except Exception as e:
        print(f"DB Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_env())
