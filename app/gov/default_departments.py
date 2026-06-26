"""
Default city departments — single source of truth.

When EAiSER provisions a city's super-admin account we pre-create a standard
set of departments so reports route to the right team from day one (the admin
can rename / delete / re-map them afterwards). Each department owns a list of
canonical AI `issue_type` values (the exact keys the classifier emits — see
app/data/issue_category_map.json), so the gov portal can show every report to
the correct department instead of relying on hand-guessed name→type maps.

Keep `issue_types` in canonical snake_case. Every type the AI can emit should
belong to exactly one default department so nothing is orphaned.
"""
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# name → (color, description, [canonical issue_types])
DEFAULT_DEPARTMENTS = [
    {
        "name": "Public Works",
        "icon": "#D4A017",
        "description": "Roads, sidewalks, drainage, trees and public infrastructure.",
        "issue_types": [
            "pothole", "road_damage", "road_debris", "damaged_sidewalk",
            "damaged_traffic_sign", "fallen_tree", "clogged_drain",
            "open_manhole", "park_playground_damage",
        ],
    },
    {
        "name": "Sanitation",
        "icon": "#F97316",
        "description": "Garbage, illegal dumping, dead animals and waste pickup.",
        "issue_types": ["garbage", "dead_animal"],
    },
    {
        "name": "Water Services",
        "icon": "#3B82F6",
        "description": "Water leaks, flooding and stormwater.",
        "issue_types": ["water_leakage", "flooding", "active_flooding"],
    },
    {
        "name": "Electrical & Traffic",
        "icon": "#EAB308",
        "description": "Streetlights, traffic signals and power lines.",
        "issue_types": ["broken_streetlight", "broken_traffic_signal", "downed_power_line"],
    },
    {
        "name": "Code Enforcement",
        "icon": "#A855F7",
        "description": "Graffiti, vandalism and abandoned vehicles.",
        "issue_types": ["graffiti_vandalism", "abandoned_vehicle"],
    },
    {
        "name": "Fire & Emergency",
        "icon": "#EF4444",
        "description": "Fire, gas leaks, hazmat and life-safety emergencies.",
        "issue_types": [
            "fire", "fire_hazard", "structural_collapse",
            "hazmat_spill", "major_gas_leak", "person_in_distress",
        ],
    },
    {
        "name": "Police",
        "icon": "#22C55E",
        "description": "Traffic accidents and public-safety incidents.",
        "issue_types": ["car_accident"],
    },
]


# Common existing/legacy department names → canonical default name. Lets us
# recognise a city's hand-made "FIRE DEPARTMENT" as the "Fire & Emergency"
# default and just back-fill its issue_types instead of creating a duplicate.
_NAME_ALIASES = {
    "public works": "Public Works",
    "public department": "Public Works",
    "public dept": "Public Works",
    "public": "Public Works",
    "transportation": "Public Works",
    "sanitation": "Sanitation",
    "waste management": "Sanitation",
    "water services": "Water Services",
    "water & utilities": "Water Services",
    "water and utilities": "Water Services",
    "water department": "Water Services",
    "water": "Water Services",
    "electrical & traffic": "Electrical & Traffic",
    "electrical and traffic": "Electrical & Traffic",
    "electrical": "Electrical & Traffic",
    "utilities": "Electrical & Traffic",
    "code enforcement": "Code Enforcement",
    "codes": "Code Enforcement",
    "fire & emergency": "Fire & Emergency",
    "fire and emergency": "Fire & Emergency",
    "fire department": "Fire & Emergency",
    "fire": "Fire & Emergency",
    "police": "Police",
    "police command": "Police",
    "police department": "Police",
}


def canonical_default_name(name: str):
    """Map an arbitrary department name to the canonical default it represents, else None."""
    key = (name or "").strip().lower()
    for d in DEFAULT_DEPARTMENTS:
        if d["name"].lower() == key:
            return d["name"]
    return _NAME_ALIASES.get(key)


def _default_by_name(name: str):
    for d in DEFAULT_DEPARTMENTS:
        if d["name"] == name:
            return d
    return None


def issue_types_for_department(name: str):
    """Canonical issue_types owned by a default department name/alias (case-insensitive). [] if unknown."""
    canon = canonical_default_name(name)
    d = _default_by_name(canon) if canon else None
    return list(d["issue_types"]) if d else []


async def seed_default_departments(db, city: str, created_by: str = "system") -> list:
    """
    Idempotently ensure the standard departments exist for `city` in the
    `gov_departments` collection (the collection the gov portal reads).

    - An existing department that maps (by name or alias) to a default — e.g. a
      hand-made "FIRE DEPARTMENT" → "Fire & Emergency" — is kept as-is; only its
      `issue_types` are back-filled if empty. No duplicate is created.
    - Defaults not represented by any existing department are created.

    Returns the list of department names that were created or back-filled.
    """
    if not city:
        logger.warning("seed_default_departments called without a city — skipping.")
        return []

    existing = await db["gov_departments"].find({"city": city}).to_list(length=500)

    # Which default each existing department already covers (name or alias).
    covered = set()
    touched = []
    for ex in existing:
        canon = canonical_default_name(ex.get("name"))
        if not canon:
            continue
        covered.add(canon)
        if not ex.get("issue_types"):
            # Existing dept (incl. legacy custom-named) — back-fill its routing.
            await db["gov_departments"].update_one(
                {"_id": ex["_id"]},
                {"$set": {"issue_types": issue_types_for_department(canon),
                          "updated_at": datetime.utcnow()}},
            )
            touched.append(ex.get("name"))

    # Create only the defaults nobody covers yet.
    for d in DEFAULT_DEPARTMENTS:
        if d["name"] in covered:
            continue
        await db["gov_departments"].insert_one({
            "name": d["name"],
            "description": d["description"],
            "icon": d["icon"],
            "issue_types": list(d["issue_types"]),
            "city": city,
            "created_at": datetime.utcnow(),
            "created_by": created_by,
            "is_default": True,
        })
        touched.append(d["name"])

    if touched:
        logger.info(f"🏛️ Seeded/updated default departments for city '{city}': {touched}")
    return touched
