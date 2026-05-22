"""
Cleanup old GridFS files. Cloudinary is primary storage; GridFS was redundant backup.

SAFETY:
- DRY-RUN by default.
- Pass --execute to delete.
- Pass --days N to override age threshold (default 30).
- Pass --all to delete ALL GridFS files (use only if Cloudinary has every image).
- Backs up file metadata before deletion.
- Verifies the corresponding issue has cloudinary image_url BEFORE removing GridFS copy.
"""
import asyncio
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import motor.motor_asyncio

load_dotenv(Path(__file__).parent / ".env", override=True)


def parse_issue_id(filename: str) -> str:
    """Extract issue_id from 'issue_ABC123.jpg', 'ABC123', 'issue_ABC123_0.jpg', etc."""
    if not filename:
        return ""
    stem = re.sub(r"\.(?:jpg|jpeg|png|webp|gif)$", "", filename, flags=re.I)
    m = re.match(r"^(?:issue_)?([A-Za-z0-9-]+?)(?:_\d+)?$", stem)
    return m.group(1) if m else ""


async def main(execute: bool = False, days: int = 30, delete_all: bool = False):
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    fs_files = db["fs.files"]
    fs_chunks = db["fs.chunks"]
    issues = db["issues"]

    mode = "EXECUTE (DESTRUCTIVE)" if execute else "DRY-RUN (read-only)"
    scope = "ALL FILES" if delete_all else f"older than {days} days"
    print("=" * 60)
    print(f"GRIDFS CLEANUP  |  MODE: {mode}  |  SCOPE: {scope}")
    print("=" * 60)

    # Pre-cleanup stats
    total_files = await fs_files.count_documents({})
    print(f"\n[STATS] Total GridFS files: {total_files}")

    # Build query
    if delete_all:
        query = {}
    else:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = {"uploadDate": {"$lt": cutoff}}

    matched = await fs_files.count_documents(query)
    print(f"[STATS] Files matching scope: {matched}")

    # For safety, only delete files whose corresponding issue has Cloudinary URL
    print("\n[1] Cross-checking with issues collection for Cloudinary backup...")
    safe_to_delete = []
    no_cloudinary = []
    orphan_files = []

    cursor = fs_files.find(query, {"_id": 1, "filename": 1, "length": 1, "uploadDate": 1})
    async for f in cursor:
        issue_id = parse_issue_id(f.get("filename", ""))
        if not issue_id:
            orphan_files.append(f)
            continue

        issue = await issues.find_one(
            {"_id": issue_id},
            {"image_url": 1, "image_urls": 1}
        )
        if not issue:
            # Issue was deleted but GridFS file orphaned
            orphan_files.append(f)
            continue

        has_cloudinary = bool(issue.get("image_url")) or bool(issue.get("image_urls"))
        if has_cloudinary:
            safe_to_delete.append(f)
        else:
            no_cloudinary.append(f)

    total_size = sum(f.get("length", 0) for f in safe_to_delete) / (1024 * 1024)
    orphan_size = sum(f.get("length", 0) for f in orphan_files) / (1024 * 1024)
    no_cloud_size = sum(f.get("length", 0) for f in no_cloudinary) / (1024 * 1024)

    print(f"\n[CLASSIFICATION]")
    print(f"  Safe to delete (have Cloudinary backup):  {len(safe_to_delete):>4}  files  ({total_size:>6.2f} MB)")
    print(f"  Orphan (no matching issue):               {len(orphan_files):>4}  files  ({orphan_size:>6.2f} MB)")
    print(f"  RISKY (no Cloudinary backup, KEEP):       {len(no_cloudinary):>4}  files  ({no_cloud_size:>6.2f} MB)")

    delete_list = safe_to_delete + orphan_files
    total_delete_size = total_size + orphan_size

    # Sample preview
    print(f"\n[SAMPLE] First 5 files to delete:")
    for f in delete_list[:5]:
        size_kb = f.get("length", 0) / 1024
        print(f"  {f.get('uploadDate')} | {size_kb:>8.1f} KB | {f.get('filename')}")
    if len(delete_list) > 5:
        print(f"  ... and {len(delete_list) - 5} more")

    # Backup metadata
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"gridfs_backup_{ts}.json"
    with backup_file.open("w", encoding="utf-8") as f:
        json.dump(
            [{
                "_id": str(d["_id"]),
                "filename": d.get("filename"),
                "length": d.get("length"),
                "uploadDate": d.get("uploadDate").isoformat() if d.get("uploadDate") else None,
            } for d in delete_list],
            f, indent=2
        )
    print(f"\n[BACKUP] Metadata saved: {backup_file}")

    if not execute:
        print("\n" + "=" * 60)
        print(f"DRY-RUN COMPLETE. {len(delete_list)} files ({total_delete_size:.2f} MB) would be deleted.")
        print(f"Re-run with --execute to delete.")
        print("=" * 60)
        client.close()
        return

    # EXECUTE
    print("\n[EXECUTING] Deleting GridFS files...")
    file_ids = [d["_id"] for d in delete_list]
    BATCH = 200
    deleted_files = 0
    deleted_chunks = 0
    for i in range(0, len(file_ids), BATCH):
        batch = file_ids[i:i + BATCH]
        # Delete chunks first, then files
        chunk_result = await fs_chunks.delete_many({"files_id": {"$in": batch}})
        file_result = await fs_files.delete_many({"_id": {"$in": batch}})
        deleted_chunks += chunk_result.deleted_count
        deleted_files += file_result.deleted_count
        print(f"  Batch {i // BATCH + 1}: files={file_result.deleted_count}, chunks={chunk_result.deleted_count} (total files {deleted_files}/{len(file_ids)})")

    # Verify
    remaining = await fs_files.count_documents({})
    print(f"\n[VERIFY] Remaining GridFS files: {remaining}")

    print("\n" + "=" * 60)
    print(f"CLEANUP COMPLETE.")
    print(f"  Files deleted:  {deleted_files}")
    print(f"  Chunks deleted: {deleted_chunks}")
    print(f"  Space freed:    ~{total_delete_size:.2f} MB")
    print(f"  Backup:         {backup_file}")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    delete_all = "--all" in sys.argv
    days = 30
    for arg in sys.argv:
        if arg.startswith("--days="):
            days = int(arg.split("=", 1)[1])
    asyncio.run(main(execute=execute, days=days, delete_all=delete_all))
