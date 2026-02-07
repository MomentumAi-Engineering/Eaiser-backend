
import asyncio
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

async def debug_issues():
    # Use the same imports as the app to ensure consistent behavior
    from services.mongodb_optimized_service import get_optimized_mongodb_service
    
    mongo_service = await get_optimized_mongodb_service()
    if not mongo_service:
        print("Failed to connect to DB")
        return

    collection = await mongo_service.get_collection("issues")
    
    # 1. Check for issues that have 'admin_review' but are 'needs_review'
    print("\n--- Issues with admin_review but status='needs_review' ---")
    query = {
        "status": "needs_review",
        "admin_review": {"$exists": True}
    }
    count = await collection.count_documents(query)
    print(f"Count: {count}")
    async for issue in collection.find(query).limit(5):
        print(f"ID: {issue['_id']}, Status: {issue.get('status')}, Review: {issue.get('admin_review')}")

    # 2. Check explicitly for the query used in get_resolved_reviews
    print("\n--- Test get_resolved_reviews query ---")
    # Simulate an admin ID. I'll pick one from the found issues if any, or a dummy one.
    # Since I don't know the exact admin ID, I'll just check distinct admin_ids in admin_review
    distinct_admins = await collection.distinct("admin_review.admin_id")
    print(f"Distinct Admin IDs in reviews: {distinct_admins}")
    
    for admin_id in distinct_admins:
        print(f"\nChecking for Admin ID: {admin_id}")
        or_conds = [{"admin_review.admin_id": str(admin_id)}]
        try:
            or_conds.append({"admin_review.admin_id": ObjectId(str(admin_id))})
        except:
            pass
            
        test_query = {
            "status": {"$in": ["rejected", "declined", "completed", "submitted", "resolved", "approved"]},
            "$or": or_conds
        }
        
        # Also run the query WITHOUT the status filter to see if "needs_review" items originate from here
        loose_query = {
             "$or": or_conds
        }
        
        matched_strict = await collection.count_documents(test_query)
        matched_loose = await collection.count_documents(loose_query)
        
        print(f"  Query match count (Strict Status): {matched_strict}")
        print(f"  Query match count (Any Status): {matched_loose}")
        
        if matched_loose > matched_strict:
            print("  !! WARNING: Some issues taken by this admin are NOT in resolved status !!")
            # Dump one such issue
            diff_query = {
                "$or": or_conds,
                "status": {"$nin": ["rejected", "declined", "completed", "submitted", "resolved", "approved"]}
            }
            async for bad_issue in collection.find(diff_query).limit(1):
                print(f"  Example Problem Issue: {bad_issue['_id']}, Status: {bad_issue.get('status')}")

if __name__ == "__main__":
    asyncio.run(debug_issues())
