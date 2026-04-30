"""V4 Tables & Reconciler Test"""
import sys
sys.path.insert(0, '.')
from utils.v4_tables import (
    reconcile_and_enrich, TIER_MAP, EMERGENCY_ADVISORIES,
    get_tier, get_priority_for_issue, is_emergency, determine_scenario
)

print(f"TIER_MAP: {len(TIER_MAP)} entries")
print(f"EMERGENCY_ADVISORIES: {len(EMERGENCY_ADVISORIES)} entries")
print()

# Test 1: Fire Hazard (Tier 0) - should get Critical priority + emergency
r = reconcile_and_enrich('fire_hazard', 'Medium', 90.0, 'safety', 'Medium')
print(f"[1] Fire Hazard: tier={r['tier']}, priority={r['priority']}, emergency={r['emergency_911']}, scenario={r['scenario']}")
assert r['tier'] == 0
assert r['priority'] == 'Critical'
assert r['emergency_911'] == True
assert r['scenario'] == 'F'
assert r['emergency_advisory'] is not None
print("    Advisory: " + r['emergency_advisory'][:60] + "...")
print()

# Test 2: Pothole (Tier 2) - should get Medium priority, no emergency
r2 = reconcile_and_enrich('pothole', 'High', 95.0, 'infrastructure', 'High')
print(f"[2] Pothole: tier={r2['tier']}, priority={r2['priority']}, emergency={r2['emergency_911']}, scenario={r2['scenario']}")
assert r2['tier'] == 2
assert r2['priority'] == 'Medium'
assert r2['emergency_911'] == False
assert r2['scenario'] == 'A'
print()

# Test 3: Unknown issue - should clamp confidence
r3 = reconcile_and_enrich('unknown_thing', 'Medium', 85.0, 'public', 'Medium')
print(f"[3] Unknown: tier={r3['tier']}, priority={r3['priority']}, emergency={r3['emergency_911']}, confidence={r3['confidence']}")
assert r3['tier'] == -1
assert r3['confidence'] == 70.0  # Clamped from 85
print()

# Test 4: Downed Power Line (Tier 0)
r4 = reconcile_and_enrich('downed_power_line', 'Low', 80.0, 'safety', 'Low')
print(f"[4] Downed Power Line: tier={r4['tier']}, priority={r4['priority']}, emergency={r4['emergency_911']}")
assert r4['tier'] == 0
assert r4['priority'] == 'Critical'
assert r4['emergency_911'] == True
print()

# Test 5: Garbage (Tier 3) - should get Low priority
r5 = reconcile_and_enrich('garbage', 'High', 95.0, 'public', 'High')
print(f"[5] Garbage: tier={r5['tier']}, priority={r5['priority']}, severity={r5['severity']}")
assert r5['tier'] == 3
assert r5['priority'] == 'Low'
assert r5['severity'] == 'Low'
print()

# Test 6: Flooding (Tier 1) - should get High priority
r6 = reconcile_and_enrich('flooding', 'Medium', 90.0, 'safety', 'Medium')
print(f"[6] Flooding: tier={r6['tier']}, priority={r6['priority']}")
assert r6['tier'] == 1
assert r6['priority'] == 'High'
print()

# Test 7: Scenario determination
s1, rev1, em1 = determine_scenario([{"issue": "pothole"}], [])
print(f"[7a] pothole only: scenario={s1}, review={rev1}, emergency={em1}")
assert s1 == 'A' and rev1 == False

s2, rev2, em2 = determine_scenario([{"issue": "pothole"}], [{"issue": "Something Weird"}])
print(f"[7b] pothole + unknown: scenario={s2}, review={rev2}")
assert s2 == 'B' and rev2 == True

s3, rev3, em3 = determine_scenario([{"issue": "fire_hazard"}, {"issue": "pothole"}], [])
print(f"[7c] fire_hazard + pothole: scenario={s3}, review={rev3}, emergency={em3}")
assert s3 == 'F' and rev3 == True and em3 == True

s4, rev4, em4 = determine_scenario([{"issue": "flooding"}, {"issue": "pothole"}], [])
print(f"[7d] flooding + pothole: scenario={s4}, review={rev4}")
assert s4 == 'C' and rev4 == False

s5, rev5, em5 = determine_scenario([], [{"issue": "Something"}])
print(f"[7e] unknown only: scenario={s5}, review={rev5}")
assert s5 == 'E' and rev5 == True

print()
print("=" * 50)
print("ALL TESTS PASSED!")
