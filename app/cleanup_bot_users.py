"""
Cleanup bot users from the April 28-29, 2026 attack.

SAFETY:
- Runs in DRY-RUN mode by default (no deletions).
- Pass --execute to actually delete.
- Excludes ANY user who has posted at least one issue (extra safety).
- Excludes admins, OAuth users (google/apple), and users with welcome_email_sent=True.
- Backs up matched IDs to a file before deletion.
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import motor.motor_asyncio

load_dotenv(Path(__file__).parent / ".env", override=True)


async def main(execute: bool = False):
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    users = db["users"]
    issues = db["issues"]

    mode = "EXECUTE (DESTRUCTIVE)" if execute else "DRY-RUN (read-only)"
    print("=" * 60)
    print(f"BOT USER CLEANUP  |  MODE: {mode}")
    print("=" * 60)

    # Step 1: collect emails of users who have posted issues (safety list)
    print("\n[1] Building protection list (users with activity)...")
    active_pipeline = [{"$group": {"_id": "$user_email"}}]
    active_results = await issues.aggregate(active_pipeline).to_list(None)
    active_emails = {r["_id"] for r in active_results if r["_id"]}
    print(f"    Protected (have posted issues): {len(active_emails)}")

    # Step 2: build the bot-match query
    # Bot signature: auth_provider=None + welcome_email_sent=None + signed up Apr 28-29, 2026
    bot_query = {
        "auth_provider": {"$in": [None]},
        "welcome_email_sent": {"$in": [None]},
        "created_at": {
            "$gte": datetime(2026, 4, 28, 0, 0, 0),
            "$lt": datetime(2026, 4, 30, 0, 0, 0),
        },
        "role": "user",
    }

    matched = await users.count_documents(bot_query)
    print(f"\n[2] Matched bot pattern: {matched:,}")
    print(f"    Pattern: auth_provider=None, welcome_email_sent=None,")
    print(f"             created between 2026-04-28 and 2026-04-30, role=user")

    # Step 3: filter out anyone in protected list (extra safety)
    print("\n[3] Filtering protected (active) users from match...")
    to_delete = []
    cursor = users.find(bot_query, {"_id": 1, "email": 1, "created_at": 1})
    async for u in cursor:
        if u.get("email") in active_emails:
            print(f"    [SKIP - has activity] {u.get('email')}")
            continue
        to_delete.append({
            "_id": u["_id"],
            "email": u.get("email"),
            "created_at": u.get("created_at"),
        })

    print(f"\n    Final delete count after safety filter: {len(to_delete):,}")

    # Step 4: write backup
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"bot_users_backup_{ts}.json"
    with backup_file.open("w", encoding="utf-8") as f:
        json.dump(
            [{"_id": str(d["_id"]), "email": d["email"], "created_at": d["created_at"].isoformat() if d["created_at"] else None} for d in to_delete],
            f, indent=2
        )
    print(f"\n[4] Backup written: {backup_file}  ({len(to_delete)} ids)")

    # Step 5: sample preview
    print("\n[5] Sample 5 users to be deleted:")
    for u in to_delete[:5]:
        print(f"    {u['email']:<40} created={u['created_at']}")
    if len(to_delete) > 5:
        print(f"    ... and {len(to_delete) - 5:,} more")

    # Step 6: execute or stop
    if not execute:
        print("\n" + "=" * 60)
        print("DRY-RUN COMPLETE. No data deleted.")
        print(f"Re-run with --execute to delete {len(to_delete):,} bot users.")
        print("=" * 60)
        client.close()
        return

    # EXECUTION PATH
    print("\n[6] EXECUTING DELETION...")
    ids = [d["_id"] for d in to_delete]
    BATCH = 5000
    deleted = 0
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        result = await users.delete_many({"_id": {"$in": batch}})
        deleted += result.deleted_count
        print(f"    Batch {i // BATCH + 1}: deleted {result.deleted_count} (total {deleted:,}/{len(ids):,})")

    # Final verification
    remaining = await users.count_documents({})
    print(f"\n[VERIFY] Users remaining in DB: {remaining:,}")

    print("\n" + "=" * 60)
    print(f"CLEANUP COMPLETE. Deleted: {deleted:,}")
    print(f"Backup at: {backup_file}")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    asyncio.run(main(execute=execute))
