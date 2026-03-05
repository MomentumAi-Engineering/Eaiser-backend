
import asyncio
from services.mongodb_optimized_service import init_optimized_mongodb, get_optimized_mongodb_service
from bson.objectid import ObjectId

async def check_types():
    await init_optimized_mongodb()
    service = await get_optimized_mongodb_service()
    
    issues_col = await service.get_collection('issues', read_only=True)
    issue = await issues_col.find_one({"image_id": {"$exists": True}})
    
    if issue:
        print(f"Issue _id type: {type(issue['_id'])}")
        print(f"Issue _id value: {issue['_id']}")
        print(f"image_id type: {type(issue['image_id'])}")
        print(f"image_id value: {issue['image_id']}")
    else:
        print("No issue with image_id found.")

if __name__ == "__main__":
    asyncio.run(check_types())
