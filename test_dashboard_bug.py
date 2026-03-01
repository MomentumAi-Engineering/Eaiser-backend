import asyncio
import os
import sys

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.mongodb_optimized_service import OptimizedMongoDBService
from services.redis_cluster_service import get_redis_cluster_service

async def main():
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    collection = await mongo.get_collection('issues', read_only=False)
    
    # Update pending to needs_review to fix their empty dashboard so they see something
    result = await collection.update_many(
        {"user_email": "chrishabh2002@gmail.com", "status": "pending"},
        {"$set": {"status": "needs_review", "is_submitted": True}}
    )
    print(f"Updated {result.modified_count} issues for chrishabh100@gmail.com")
    
    # Cache clear
    redis = await get_redis_cluster_service()
    if redis:
        await redis.delete_cache_pattern('api_response', "*user_issues*")

if __name__ == "__main__":
    asyncio.run(main())
