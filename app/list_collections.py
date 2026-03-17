import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())
load_dotenv()

from services.mongodb_optimized_service import get_optimized_mongodb_service

async def main():
    try:
        mongo = await get_optimized_mongodb_service()
        if not mongo:
            print("Failed to connect to MongoDB")
            return
        collections = await mongo.db.list_collection_names()
        print("Collections:", collections)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
