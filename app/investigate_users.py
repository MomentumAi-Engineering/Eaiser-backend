"""
READ-ONLY investigation of 121K users.
NO deletions. Just findings.
"""
import asyncio
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import motor.motor_asyncio

load_dotenv(Path(__file__).parent / ".env", override=True)


async def main():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    users = db["users"]

    print("=" * 60)
    print("USERS INVESTIGATION (READ-ONLY)")
    print("=" * 60)

    total = await users.count_documents({})
    print(f"Total users: {total:,}\n")

    # By auth_provider
    print("[1] BY AUTH PROVIDER")
    pipeline = [{"$group": {"_id": "$auth_provider", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id'] or 'none'):<30} {r['count']:>8,}")

    # By role
    print("\n[2] BY ROLE")
    pipeline = [{"$group": {"_id": "$role", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id'] or 'none'):<30} {r['count']:>8,}")

    # By email_verified
    print("\n[3] BY email_verified")
    pipeline = [{"$group": {"_id": "$email_verified", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id']):<30} {r['count']:>8,}")

    # By welcome_email_sent
    print("\n[4] BY welcome_email_sent")
    pipeline = [{"$group": {"_id": "$welcome_email_sent", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id']):<30} {r['count']:>8,}")

    # By is_active
    print("\n[5] BY is_active")
    pipeline = [{"$group": {"_id": "$is_active", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id']):<30} {r['count']:>8,}")

    # Created at distribution
    print("\n[6] SIGNUP TIMELINE (by month)")
    pipeline = [
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": -1}},
        {"$limit": 15}
    ]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id'] or 'unknown'):<30} {r['count']:>8,}")

    # Email domain breakdown
    print("\n[7] TOP EMAIL DOMAINS")
    pipeline = [
        {"$match": {"email": {"$type": "string", "$ne": ""}}},
        {"$addFields": {"domain": {"$arrayElemAt": [{"$split": ["$email", "@"]}, 1]}}},
        {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    for r in await users.aggregate(pipeline).to_list(None):
        print(f"  {str(r['_id'] or 'none'):<30} {r['count']:>8,}")

    # Sample suspicious users
    print("\n[8] SAMPLE USERS (first 5)")
    sample = await users.find({}, {"email": 1, "auth_provider": 1, "created_at": 1, "is_active": 1, "_id": 0}).limit(5).to_list(None)
    for u in sample:
        print(f"  {u}")

    # Recent signups
    print("\n[9] LATEST 5 SIGNUPS")
    latest = await users.find().sort("created_at", -1).limit(5).to_list(None)
    for u in latest:
        print(f"  email={u.get('email')} | provider={u.get('auth_provider')} | created={u.get('created_at')}")

    # Burst detection — signups per day in last 30 days
    print("\n[10] DAILY SIGNUP BURSTS (last 30 days)")
    cutoff = datetime.utcnow() - timedelta(days=30)
    pipeline = [
        {"$match": {"created_at": {"$gte": cutoff}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": -1}}
    ]
    for r in await users.aggregate(pipeline).to_list(None):
        flag = "  <-- BURST!" if r['count'] > 500 else ""
        print(f"  {r['_id']:<15} {r['count']:>6}{flag}")

    client.close()
    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE (no data modified)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
