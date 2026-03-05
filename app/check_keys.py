import asyncio
import os
from dotenv import load_dotenv
import motor.motor_asyncio
import json

async def check_issues():
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI') or 'mongodb://localhost:27017'
    db_name = os.getenv('MONGODB_NAME', 'eaiser')
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    
    issues = await db.issues.find().limit(5).to_list(length=5)
    for i in issues:
        # remove big fields for clear view
        if 'report' in i: i['report'] = 'TRUNCATED'
        if 'unified_report' in i: i['unified_report'] = 'TRUNCATED'
        i['_id'] = str(i['_id'])
        print(json.dumps(i, indent=2))

if __name__ == "__main__":
    asyncio.run(check_issues())
