"""
Add missing indexes for 3-city launch (Fairview, Franklin, Chattanooga).
SAFE: createIndex is non-destructive — skips if index already exists.
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, GEOSPHERE

load_dotenv(Path(__file__).parent / ".env", override=True)


async def main():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")

    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    issues = db["issues"]

    print("=" * 60)
    print("ADDING MISSING INDEXES FOR 3-CITY LAUNCH")
    print("=" * 60)

    indexes_to_add = [
        ("city_1", [("city", ASCENDING)], "Filter issues by city (Fairview/Franklin/Chattanooga)"),
        ("created_at_-1", [("created_at", DESCENDING)], "Sort by newest first"),
        ("city_1_status_1", [("city", ASCENDING), ("status", ASCENDING)], "City + status filter"),
        ("city_1_created_at_-1", [("city", ASCENDING), ("created_at", DESCENDING)], "City + recent filter"),
        ("department_tag_1", [("department_tag", ASCENDING)], "Department routing"),
        ("location_2dsphere", [("location", GEOSPHERE)], "Geo nearby queries"),
    ]

    existing = await issues.index_information()
    print(f"\nExisting indexes: {len(existing)}")

    created = []
    skipped = []
    errors = []

    for name, keys, description in indexes_to_add:
        try:
            already_exists = False
            for ex_name, ex_info in existing.items():
                ex_keys = ex_info.get('key', [])
                if ex_keys == keys or (len(ex_keys) == len(keys) and all(a == b for a, b in zip(ex_keys, keys))):
                    print(f"  [SKIP] {name} — already exists as '{ex_name}'")
                    skipped.append(name)
                    already_exists = True
                    break

            if already_exists:
                continue

            result = await issues.create_index(keys, name=name, background=True, sparse=True)
            print(f"  [OK] Created: {name} — {description}")
            created.append(name)

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            errors.append((name, str(e)))

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(created)} created, {len(skipped)} skipped, {len(errors)} errors")
    print("=" * 60)
    if created:
        print(f"\nNew indexes created: {created}")
    if errors:
        print(f"\nErrors: {errors}")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
