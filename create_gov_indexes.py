"""
Apply ALL MongoDB indexes the app relies on (core + Gov Portal) to the SAME
database the application uses at runtime. Idempotent and resilient — a single
pre-existing/conflicting index never aborts the rest.

The same logic also runs automatically at app startup
(services.mongodb_service.create_indexes); use this for an on-demand apply
(e.g. right after deploy) without restarting.

Run:  python create_gov_indexes.py
"""
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from services import mongodb_service as m  # noqa: E402


async def main():
    await m.init_db()
    await m.create_indexes()
    db = m.db
    print(f"Applied indexes on database: {db.name}")
    for coll in ("gov_departments", "government_users", "issues"):
        idx = sorted((await db[coll].index_information()).keys())
        print(f"  {coll}: {idx}")


if __name__ == "__main__":
    asyncio.run(main())
