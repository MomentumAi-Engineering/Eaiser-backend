from services.mongodb_service import get_db
import asyncio

async def check():
    db = await get_db()
    count = await db['issues'].count_documents({})
    print(f"Total issues: {count}")
    
    cursor = db['issues'].find({}).limit(5)
    issues = await cursor.to_list(length=5)
    for i, issue in enumerate(issues):
        print(f"Issue {i}: {issue.get('issue_type')} - {issue.get('status')} - Assigned to: {issue.get('assigned_to')}")
        
    counts = await db['government_users'].count_documents({})
    print(f"Total government_users: {counts}")
    cursor = db['government_users'].find({}).limit(5)
    users = await cursor.to_list(length=5)
    for u in users:
        print(f"User: {u.get('name')} - {u.get('email')} - {u.get('dept')} - {u.get('zip_code')}")

if __name__ == "__main__":
    asyncio.run(check())
