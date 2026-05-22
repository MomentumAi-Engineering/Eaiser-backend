"""
Safe DB audit — analyzes GridFS, users, and indexes.
READ-ONLY. No deletions. Reports findings for manual decision.
"""
import asyncio
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import motor.motor_asyncio

load_dotenv(Path(__file__).parent / ".env", override=True)


async def audit_gridfs(db):
    print("\n" + "=" * 60)
    print("[1] GRIDFS AUDIT (fs.files + fs.chunks)")
    print("=" * 60)

    total_files = await db["fs.files"].count_documents({})
    print(f"Total files in GridFS: {total_files}")

    if total_files == 0:
        print("  No GridFS files — chunks may be orphans.")
        return

    # Group by upload date
    pipeline = [
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m", "date": "$uploadDate"}},
            "count": {"$sum": 1},
            "total_size": {"$sum": "$length"}
        }},
        {"$sort": {"_id": -1}}
    ]
    by_month = await db["fs.files"].aggregate(pipeline).to_list(None)
    print("\nFiles by month:")
    print(f"  {'Month':<12} {'Files':>8} {'Size(MB)':>12}")
    for m in by_month:
        size_mb = (m['total_size'] or 0) / (1024 * 1024)
        print(f"  {m['_id'] or 'unknown':<12} {m['count']:>8} {size_mb:>12.2f}")

    # Group by content type
    pipeline = [
        {"$group": {
            "_id": "$contentType",
            "count": {"$sum": 1},
            "total_size": {"$sum": "$length"}
        }},
        {"$sort": {"total_size": -1}}
    ]
    by_type = await db["fs.files"].aggregate(pipeline).to_list(None)
    print("\nFiles by type:")
    print(f"  {'Type':<30} {'Count':>8} {'Size(MB)':>12}")
    for t in by_type:
        size_mb = (t['total_size'] or 0) / (1024 * 1024)
        print(f"  {str(t['_id'])[:30]:<30} {t['count']:>8} {size_mb:>12.2f}")

    # Sample 5 oldest files
    print("\nOldest 5 files (delete candidates):")
    oldest = await db["fs.files"].find().sort("uploadDate", 1).limit(5).to_list(None)
    for f in oldest:
        size_kb = (f.get('length', 0)) / 1024
        upload = f.get('uploadDate', 'unknown')
        name = f.get('filename', 'unnamed')[:40]
        print(f"  {upload} | {size_kb:>8.1f} KB | {name}")

    # Recent files (might be in use)
    cutoff = datetime.utcnow() - timedelta(days=30)
    recent = await db["fs.files"].count_documents({"uploadDate": {"$gte": cutoff}})
    old = total_files - recent
    print(f"\nSummary:")
    print(f"  Files uploaded in last 30 days: {recent}  (likely in use)")
    print(f"  Files older than 30 days:       {old}  (CLEANUP CANDIDATES)")


async def audit_users(db):
    print("\n" + "=" * 60)
    print("[2] USERS AUDIT")
    print("=" * 60)

    total = await db["users"].count_documents({})
    print(f"Total users: {total:,}")

    # Sample fields
    sample = await db["users"].find_one()
    if sample:
        print(f"Sample user fields: {list(sample.keys())[:15]}")

    # Test email patterns
    test_patterns = [
        ("test@", {"email": re.compile(r"test.*@", re.I)}),
        ("@test", {"email": re.compile(r"@test\.", re.I)}),
        ("example.com", {"email": re.compile(r"@example\.com$", re.I)}),
        ("@mailinator", {"email": re.compile(r"@mailinator", re.I)}),
        ("temp/dummy", {"email": re.compile(r"^(temp|dummy|fake)", re.I)}),
        ("no email", {"email": {"$in": [None, ""]}}),
    ]
    print("\nTest/fake user detection:")
    print(f"  {'Pattern':<20} {'Count':>10}")
    for label, query in test_patterns:
        try:
            count = await db["users"].count_documents(query)
            print(f"  {label:<20} {count:>10,}")
        except Exception:
            pass

    # By creation date if available
    try:
        cutoff_old = datetime.utcnow() - timedelta(days=365)
        very_old = await db["users"].count_documents({"created_at": {"$lt": cutoff_old}})
        print(f"\n  Users created > 1 year ago: {very_old:,}")
    except Exception:
        pass

    # Users with NO activity (no issues posted)
    print("\nChecking active vs inactive users...")
    active_pipeline = [
        {"$group": {"_id": "$user_email"}}
    ]
    active_emails = await db["issues"].aggregate(active_pipeline).to_list(None)
    active_count = len(active_emails)
    print(f"  Users who posted at least 1 issue: {active_count:,}")
    print(f"  Users with ZERO activity:         {total - active_count:,}  (potential cleanup)")


async def audit_indexes(db):
    print("\n" + "=" * 60)
    print("[3] INDEX AUDIT (issues collection)")
    print("=" * 60)

    indexes = await db["issues"].index_information()
    print(f"Total indexes on 'issues': {len(indexes)}")
    for name, info in indexes.items():
        keys = info.get('key', [])
        keys_str = ", ".join([f"{k[0]}:{k[1]}" for k in keys])
        print(f"  {name:<40} -> {keys_str}")

    # Critical query checks for 3-city launch
    print("\nKey indexes needed for 3-city launch:")
    needed = ['zip_code', 'city', 'created_at', 'user_email', 'status', 'department_tag']
    indexed_fields = set()
    for info in indexes.values():
        for key in info.get('key', []):
            indexed_fields.add(key[0])

    for field in needed:
        status = "[OK]" if field in indexed_fields else "[MISSING]"
        print(f"  {status} {field}")

    # Check if geo index exists for location queries
    has_geo = any(
        any(k[1] in ('2dsphere', '2d') for k in info.get('key', []))
        for info in indexes.values()
    )
    print(f"  {'[OK]' if has_geo else '[MISSING]'} geospatial index (for nearby reports)")

    # City distribution check (for 3-city launch readiness)
    print("\nCurrent city distribution in 'issues':")
    city_pipeline = [
        {"$group": {"_id": "$city", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    cities = await db["issues"].aggregate(city_pipeline).to_list(None)
    for c in cities:
        print(f"  {str(c['_id'] or 'unknown')[:30]:<30} {c['count']:>6}")


async def main():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")

    if not uri:
        print("[ERROR] MONGO_URI not found")
        return

    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]

    print("=" * 60)
    print(f"DB AUDIT (READ-ONLY) — DB: {db_name}")
    print("=" * 60)

    await audit_gridfs(db)
    await audit_users(db)
    await audit_indexes(db)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE — No data was modified")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
