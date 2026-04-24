"""Clear all Redis cache - one-time script"""
import asyncio

async def clear():
    try:
        from services.redis_service import get_redis_service
        svc = await get_redis_service()
        if svc and svc.redis_client:
            await svc.redis_client.flushdb()
            print("REDIS CACHE CLEARED SUCCESSFULLY")
        else:
            print("Redis not connected - no cache to clear")
    except Exception as e:
        print(f"Redis not available locally: {e}")
        print("This is OK for local dev - no cache to clear")

asyncio.run(clear())
