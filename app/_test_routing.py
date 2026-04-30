"""Quick test to verify _canonical_issue routing is correct"""
import sys
sys.path.insert(0, '.')
from utils.location import _canonical_issue

# Test all expected mappings
tests = {
    # Direct matches
    "pothole": "pothole",
    "garbage": "garbage",
    "broken_streetlight": "broken_streetlight",
    "road_damage": "road_damage",
    "graffiti_vandalism": "graffiti_vandalism",
    "damaged_sidewalk": "damaged_sidewalk",
    "damaged_traffic_sign": "damaged_traffic_sign",
    "fallen_tree": "fallen_tree",
    "clogged_drain": "clogged_drain",
    "park_playground_damage": "park_playground_damage",
    "broken_traffic_signal": "broken_traffic_signal",
    "abandoned_vehicle": "abandoned_vehicle",
    "dead_animal": "dead_animal",
    "water_leakage": "water_leakage",
    "open_manhole": "open_manhole",
    "flooding": "flooding",
    "fire": "fire",
    "police": "police",
    "car_accident": "car_accident",
    "fire_hazard": "fire_hazard",
    "downed_power_line": "downed_power_line",
    
    # AI variations that should be normalized
    "Pothole": "pothole",
    "Road Crack": "road_damage",
    "Open Manhole": "open_manhole",     # Was BUG: matched "hole" -> road_damage
    "Dead Animal on Road": "dead_animal",
    "Garbage Dump": "garbage",
    "Broken Street Light": "broken_streetlight",
    "Graffiti": "graffiti_vandalism",
    "Smoke Hazard": "fire",
    "Water Pipe Burst": "water_leakage",
    "Flood": "flooding",
    "Abandoned Car": "abandoned_vehicle",
    "Broken Sidewalk": "damaged_sidewalk",
    "Traffic Light Broken": "broken_traffic_signal",
    "Fallen Branch": "fallen_tree",
    "Park Damage": "park_playground_damage",
    "Theft": "police",
    
    # Tier 0 AI variations
    "Car Accident": "car_accident",
    "Vehicle Collision": "car_accident",
    "Vehicle Crash": "car_accident",
    "Fire Hazard": "fire_hazard",
    "Structure Fire": "fire_hazard",
    "Downed Power Line": "downed_power_line",
    "Fallen Wire": "downed_power_line",
}

print("=" * 65)
print(f"{'INPUT':<30} {'EXPECTED':<20} {'GOT':<20} {'OK?'}")
print("=" * 65)

passed = 0
failed = 0
for input_val, expected in tests.items():
    result = _canonical_issue(input_val)
    ok = result == expected
    status = "✅" if ok else "❌ FAIL"
    print(f"{input_val:<30} {expected:<20} {result:<20} {status}")
    if ok:
        passed += 1
    else:
        failed += 1

print("=" * 65)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
