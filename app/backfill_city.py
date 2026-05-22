"""
Backfill `city`, `state`, and `location` (GeoJSON) fields on existing issues.

SAFE: Only UPDATES existing docs. No deletions.
Source 1: parse address string ("Fairview, TN 37062, USA" -> "Fairview", "TN")
Source 2: ZIP code lookup for TN target cities (37062, 37064, 37403, etc.)
"""
import asyncio
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import motor.motor_asyncio

load_dotenv(Path(__file__).parent / ".env", override=True)


ZIP_TO_CITY = {
    "37062": ("Fairview", "TN"),
    "37064": ("Franklin", "TN"),
    "37067": ("Franklin", "TN"),
    "37068": ("Franklin", "TN"),
    "37069": ("Franklin", "TN"),
    "37401": ("Chattanooga", "TN"),
    "37402": ("Chattanooga", "TN"),
    "37403": ("Chattanooga", "TN"),
    "37404": ("Chattanooga", "TN"),
    "37405": ("Chattanooga", "TN"),
    "37406": ("Chattanooga", "TN"),
    "37407": ("Chattanooga", "TN"),
    "37408": ("Chattanooga", "TN"),
    "37409": ("Chattanooga", "TN"),
    "37410": ("Chattanooga", "TN"),
    "37411": ("Chattanooga", "TN"),
    "37412": ("Chattanooga", "TN"),
    "37415": ("Chattanooga", "TN"),
    "37416": ("Chattanooga", "TN"),
    "37419": ("Chattanooga", "TN"),
    "37421": ("Chattanooga", "TN"),
    "37201": ("Nashville", "TN"),
    "37203": ("Nashville", "TN"),
    "37205": ("Nashville", "TN"),
    "37207": ("Nashville", "TN"),
    "37209": ("Nashville", "TN"),
    "37211": ("Nashville", "TN"),
    "37215": ("Nashville", "TN"),
    "37221": ("Nashville", "TN"),
}


def parse_city_from_address(address: str):
    """Parse 'Fairview, TN 37062, USA' -> ('Fairview', 'TN')."""
    if not address or not isinstance(address, str):
        return "", ""
    parts = [p.strip() for p in address.split(",")]
    if not parts:
        return "", ""
    city = parts[0] if parts[0] else ""
    state = ""
    if len(parts) >= 2:
        # Second part typically "TN 37062" — extract state code
        tokens = parts[1].strip().split()
        if tokens and len(tokens[0]) == 2 and tokens[0].isalpha():
            state = tokens[0].upper()
    return city, state


async def main():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "eaiser")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri, tls=True)
    db = client[db_name]
    issues = db["issues"]

    print("=" * 60)
    print("BACKFILL: city, state, location (READ-MODIFY)")
    print("=" * 60)

    total = await issues.count_documents({})
    missing_city = await issues.count_documents({"$or": [
        {"city": {"$exists": False}},
        {"city": {"$in": [None, "", "Unknown"]}}
    ]})
    print(f"Total issues:        {total}")
    print(f"Missing city field:  {missing_city}")

    cursor = issues.find(
        {},
        {"_id": 1, "address": 1, "zip_code": 1, "latitude": 1, "longitude": 1, "city": 1, "state": 1, "location": 1}
    )

    updates = 0
    from_zip = 0
    from_address = 0
    no_data = 0
    geo_added = 0

    bulk_ops = []
    BATCH = 100

    async for doc in cursor:
        update_fields = {}

        # 1. Derive city/state
        current_city = doc.get("city")
        if not current_city or current_city == "Unknown":
            zip_code = (doc.get("zip_code") or "").strip()
            city, state = ZIP_TO_CITY.get(zip_code, ("", ""))
            source = "zip"

            if not city:
                city, state = parse_city_from_address(doc.get("address", ""))
                source = "address"

            if city:
                update_fields["city"] = city
                if state:
                    update_fields["state"] = state
                if source == "zip":
                    from_zip += 1
                else:
                    from_address += 1
            else:
                no_data += 1

        # 2. Add GeoJSON location for 2dsphere queries
        if not doc.get("location"):
            lat = doc.get("latitude")
            lon = doc.get("longitude")
            try:
                if lat and lon and float(lat) != 0 and float(lon) != 0:
                    update_fields["location"] = {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)]
                    }
                    geo_added += 1
            except (TypeError, ValueError):
                pass

        if update_fields:
            bulk_ops.append({"_id": doc["_id"], "fields": update_fields})

        if len(bulk_ops) >= BATCH:
            from pymongo import UpdateOne
            ops = [UpdateOne({"_id": op["_id"]}, {"$set": op["fields"]}) for op in bulk_ops]
            result = await issues.bulk_write(ops, ordered=False)
            updates += result.modified_count
            bulk_ops = []

    if bulk_ops:
        from pymongo import UpdateOne
        ops = [UpdateOne({"_id": op["_id"]}, {"$set": op["fields"]}) for op in bulk_ops]
        result = await issues.bulk_write(ops, ordered=False)
        updates += result.modified_count

    print("\n" + "=" * 60)
    print("BACKFILL RESULTS")
    print("=" * 60)
    print(f"Documents updated:        {updates}")
    print(f"  City from ZIP lookup:   {from_zip}")
    print(f"  City from address parse:{from_address}")
    print(f"  GeoJSON location added: {geo_added}")
    print(f"  No data available:      {no_data}")

    # Verify
    print("\n=== VERIFICATION (city distribution after backfill) ===")
    pipeline = [{"$group": {"_id": "$city", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}, {"$limit": 15}]
    cities = await issues.aggregate(pipeline).to_list(None)
    for c in cities:
        print(f"  {str(c['_id'] or '(none)')[:30]:<30} {c['count']:>6}")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
