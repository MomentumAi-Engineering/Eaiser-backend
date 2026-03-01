import asyncio
import os
import sys

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.mongodb_optimized_service import OptimizedMongoDBService

async def main():
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    
    collection = await mongo.get_collection('users', read_only=True)
    user = await collection.find_one({"email": "chrishabh100@gmail.com"})
    
    if user:
        print(f"User Name: {user.get('name')}")
        print(f"User Full Name: {user.get('full_name')}")
    else:
        print("No user found")

if __name__ == "__main__":
    asyncio.run(main())
