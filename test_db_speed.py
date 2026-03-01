import asyncio
import time
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.mongodb_optimized_service import OptimizedMongoDBService

async def main():
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    
    collection = await mongo.get_collection('issues', read_only=True)
    
    match_query = {
        "user_email": {"$in": ["test@test.com", "TEST@TEST.COM"]},
        "$or": [
            {"is_submitted": True},
            {"is_submitted": {"$exists": False}, "status": {"$ne": "pending"}}
        ]
    }
    
    print("Testing explain...")
    explanation = await collection.find(match_query).sort([('timestamp', -1)]).explain()
    winning_plan = explanation.get('queryPlanner', {}).get('winningPlan', {})
    print(json.dumps(winning_plan, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
