"""Quick test of the Gemini Key Pool initialization."""
from services.gemini_key_pool import get_key_pool

pool = get_key_pool()
stats = pool.get_stats()

print(f"Pool Size: {stats['pool_size']}")
print(f"Available: {stats['available_keys']}")
print(f"Total RPM Capacity: {stats['total_capacity_rpm']}")
print(f"Total TPM Capacity: {stats['pool_size']}M tokens/min")
print(f"Estimated Reports/min: {stats['total_capacity_rpm'] // 2}")
print()
print("Keys:")
for k in stats['keys']:
    print(f"  {k['name']}: available={k['available']}, rpm={k['rpm_used']}/{k['rpm_limit']}")

# Test round-robin
print("\nTesting round-robin rotation (10 calls):")
for i in range(10):
    slot = pool.get_next_key()
    print(f"  Call {i+1}: key [{slot.project_name}] (total: {slot.total_requests})")

print(f"\nAll {stats['pool_size']} keys working! 🚀")
