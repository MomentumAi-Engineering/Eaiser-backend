import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.mongodb_optimized_service import OptimizedMongoDBService
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure

async def main():
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    
    collection = await mongo.get_collection('issues', read_only=False)
    
    print("Creating compound index...")
    try:
        await collection.create_index([("user_email", ASCENDING), ("timestamp", DESCENDING)])
        print("user_email_timestamp index created")
    except OperationFailure as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
