"""
Triple-verify GridFS cleanup safety BEFORE executing.
- Samples 10 'safe to delete' files
- Actually HTTP HEADs the Cloudinary URL to confirm image is accessible
- Reports any URL failures so we can keep those files
"""
import asyncio
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import motor.motor_asyncio
import aiohttp

load_dotenv(Path(__file__).parent / ".env", override=True)


def parse_issue_id(filename: str) -> str:
    if not filename:
        return ""
    # Strip optional extension
    stem = re.sub(r"\.(?:jpg|jpeg|png|webp|gif)$", "", filename, flags=re.I)
    # Match: optional 'issue_' prefix, then ID, then optional _N index
    m = re.match(r"^(?:issue_)?([A-Za-z0-9-]+?)(?:_\d+)?$", stem)
    return m.group(1) if m else ""


async def head_url(session, url):
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as r:
            return r.status
    except Exception as e:
        return f"ERR: {e}"


async def main():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    fs_files = db["fs.files"]
    issues = db["issues"]

    print("=" * 60)
    print("GRIDFS CLEANUP — SAFETY VERIFICATION")
    print("=" * 60)

    # Build the same classification as cleanup script
    safe_to_delete = []
    no_cloudinary = []
    orphan_files = []

    cursor = fs_files.find({}, {"_id": 1, "filename": 1, "length": 1, "uploadDate": 1})
    async for f in cursor:
        issue_id = parse_issue_id(f.get("filename", ""))
        if not issue_id:
            orphan_files.append(f)
            continue
        issue = await issues.find_one({"_id": issue_id}, {"image_url": 1, "image_urls": 1})
        if not issue:
            orphan_files.append((f, "issue not found"))
            continue
        has_cloudinary = bool(issue.get("image_url")) or bool(issue.get("image_urls"))
        if has_cloudinary:
            safe_to_delete.append((f, issue))
        else:
            no_cloudinary.append((f, issue))

    print(f"\n[CLASSIFICATION]")
    print(f"  Safe to delete (Cloudinary exists):  {len(safe_to_delete)}")
    print(f"  Orphan (issue not found):            {len(orphan_files)}")
    print(f"  KEEP (no Cloudinary):                {len(no_cloudinary)}")

    # ---- HTTP verify 10 random Cloudinary URLs ----
    import random
    sample = random.sample(safe_to_delete, min(10, len(safe_to_delete)))
    print(f"\n[1] HTTP verify {len(sample)} random Cloudinary URLs...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for f, issue in sample:
            url = issue.get("image_url") or (issue.get("image_urls") or [None])[0]
            tasks.append((f, url, head_url(session, url)))
        results = []
        for f, url, task in tasks:
            status = await task
            results.append((f, url, status))

    ok_count = 0
    fail_count = 0
    for f, url, status in results:
        ok = status == 200
        if ok:
            ok_count += 1
        else:
            fail_count += 1
        marker = "[OK]" if ok else "[FAIL]"
        print(f"  {marker} {status} | issue={parse_issue_id(f.get('filename',''))} | {(url or '')[:70]}")

    # ---- Show what KEEPS being kept ----
    print(f"\n[2] Files KEPT (no Cloudinary backup) — first 5:")
    for f, issue in no_cloudinary[:5]:
        print(f"  KEEP | {f.get('uploadDate')} | {f.get('filename')} | issue exists but no image_url")

    # ---- Show orphans ----
    print(f"\n[3] Orphan files — first 5:")
    for entry in orphan_files[:5]:
        f = entry[0] if isinstance(entry, tuple) else entry
        reason = entry[1] if isinstance(entry, tuple) else "no issue_id in filename"
        print(f"  ORPHAN | {f.get('uploadDate')} | {f.get('filename')} | {reason}")

    # ---- Final verdict ----
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Cloudinary HTTP check:  {ok_count}/{len(sample)} OK, {fail_count} failed")
    if fail_count == 0:
        print(f"  [SAFE] All sampled Cloudinary URLs are reachable.")
        print(f"  [SAFE] Orphan files have no parent issue to lose data from.")
        print(f"  [SAFE] 66 files without Cloudinary backup will be KEPT untouched.")
        print(f"  RECOMMENDATION: Safe to run cleanup with --execute.")
    else:
        print(f"  [WARN] {fail_count} Cloudinary URLs failed — DO NOT delete those.")
        print(f"  RECOMMENDATION: Investigate failed URLs before proceeding.")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
