
import asyncio
from services.mongodb_optimized_service import init_optimized_mongodb, get_optimized_mongodb_service
from bson.objectid import ObjectId

async def check_images():
    await init_optimized_mongodb()
    service = await get_optimized_mongodb_service()
    
    issues_col = await service.get_collection('issues', read_only=True)
    recent_issues = await issues_col.find().sort('timestamp', -1).limit(10).to_list(length=10)
    
    print(f"Found {len(recent_issues)} recent issues.")
    for issue in recent_issues:
        issue_id = str(issue.get('_id'))
        image_id = issue.get('image_id')
        issue_type = issue.get('issue_type')
        image_hash = issue.get('image_hash')
        status = issue.get('status')
        print(f"ID: {issue_id} | Type: {issue_type} | Status: {status} | ImageID: {image_id}")
        
        if image_id and service.fs:
            try:
                from bson.objectid import ObjectId
                gridout = await service.fs.open_download_stream(ObjectId(image_id))
                print(f"  ✅ Image found. Size: {gridout.length} bytes")
            except Exception as e:
                print(f"  ❌ Image {image_id} NOT found: {e}")
        elif image_hash and image_hash.startswith("manual_"):
            print("  ℹ️ Manual report (no image expected)")
        else:
            print("  ⚠️ No image_id found.")

if __name__ == "__main__":
    asyncio.run(check_images())
