import asyncio
import os
import sys

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.email_service import notify_user_status_change
from services.mongodb_optimized_service import OptimizedMongoDBService

async def main():
    os.environ['EMAIL_DRY_RUN'] = 'true'
    mongo = OptimizedMongoDBService()
    await mongo.connect()
    
    collection = await mongo.get_collection('issues', read_only=True)
    issue = await collection.find_one({"is_submitted": True})
    
    if issue:
        issue_id = str(issue["_id"])
        email = issue.get("user_email", "test@test.com")
        print(f"Testing notify_user_status_change for {issue_id}")
        
        # Test Approved
        print("\n--- TEST APPROVED ---")
        await notify_user_status_change(email, issue_id, "approved", "Good job")
        
        # Test Declined
        print("\n--- TEST DECLINED ---")
        await notify_user_status_change(email, issue_id, "rejected", "Not valid")

if __name__ == "__main__":
    asyncio.run(main())
