import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.mongodb_optimized_service import OptimizedMongoDBService
from pymongo import ASCENDING, DESCENDING

async def main():
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    
    collection = await mongo.get_collection('issues', read_only=False)
    
    print("Creating specific indexes...")
    # Index for user_email
    await collection.create_index([("user_email", ASCENDING)])
    
    # Index for status
    await collection.create_index([("status", ASCENDING), ("timestamp", DESCENDING)])
    
    # Index for is_submitted
    await collection.create_index([("is_submitted", ASCENDING)])
    
    print("Indexes built.")

if __name__ == "__main__":
    asyncio.run(main())
