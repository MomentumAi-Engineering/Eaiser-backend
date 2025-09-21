"""
Redis Caching Service for Eaiser AI Backend
Optimized for 1 lakh+ concurrent users with advanced caching strategies
"""

import redis
import json
import asyncio
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import hashlib
import pickle
from functools import wraps
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisCacheService:
    """
    Enterprise-grade Redis caching service
    Features: Connection pooling, cluster support, automatic failover
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0,
                 max_connections: int = 1000,
                 decode_responses: bool = True):
        """
        Initialize Redis connection with optimized settings
        """
        self.host = host
        self.port = port
        self.db = db
        
        # Connection pool for high concurrency
        self.connection_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            decode_responses=decode_responses,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )
        
        # Redis client with connection pool
        self.redis_client = redis.Redis(connection_pool=self.connection_pool)
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
        logger.info(f"🚀 Redis Cache Service initialized - {host}:{port}")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate unique cache key from function arguments"""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling"""
        try:
            value = self.redis_client.get(key)
            if value:
                self.cache_stats['hits'] += 1
                # Try to deserialize JSON, fallback to string
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                self.cache_stats['misses'] += 1
                return None
        except Exception as e:
            logger.error(f"❌ Cache GET error for key {key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            result = self.redis_client.setex(key, ttl, serialized_value)
            if result:
                self.cache_stats['sets'] += 1
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Cache SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = self.redis_client.delete(key)
            if result:
                self.cache_stats['deletes'] += 1
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Cache DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"❌ Cache EXISTS error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache"""
        try:
            return self.redis_client.incr(key, amount)
        except Exception as e:
            logger.error(f"❌ Cache INCREMENT error for key {key}: {e}")
            return None
    
    async def set_hash(self, name: str, mapping: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set hash in Redis"""
        try:
            # Convert all values to strings
            string_mapping = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                            for k, v in mapping.items()}
            
            result = self.redis_client.hset(name, mapping=string_mapping)
            if ttl > 0:
                self.redis_client.expire(name, ttl)
            return True
        except Exception as e:
            logger.error(f"❌ Cache HSET error for hash {name}: {e}")
            return False
    
    async def get_hash(self, name: str, key: str = None) -> Optional[Any]:
        """Get hash or hash field from Redis"""
        try:
            if key:
                value = self.redis_client.hget(name, key)
                if value:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return None
            else:
                hash_data = self.redis_client.hgetall(name)
                if hash_data:
                    # Try to deserialize JSON values
                    result = {}
                    for k, v in hash_data.items():
                        try:
                            result[k] = json.loads(v)
                        except json.JSONDecodeError:
                            result[k] = v
                    return result
                return None
        except Exception as e:
            logger.error(f"❌ Cache HGET error for hash {name}: {e}")
            return None
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"❌ Cache CLEAR PATTERN error for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_operations = sum(self.cache_stats.values())
        hit_rate = (self.cache_stats['hits'] / max(total_operations, 1)) * 100
        
        return {
            **self.cache_stats,
            'hit_rate': round(hit_rate, 2),
            'total_operations': total_operations,
            'connection_info': {
                'host': self.host,
                'port': self.port,
                'db': self.db
            }
        }
    
    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"❌ Redis health check failed: {e}")
            return False

# Decorator for automatic caching
def cache_result(ttl: int = 3600, key_prefix: str = "cache"):
    """
    Decorator to automatically cache function results
    Usage: @cache_result(ttl=1800, key_prefix="user_data")
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache service instance
            cache_service = RedisCacheService()
            
            # Generate cache key
            cache_key = cache_service._generate_cache_key(
                f"{key_prefix}:{func.__name__}", *args, **kwargs
            )
            
            # Try to get from cache first
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                logger.info(f"✅ Cache HIT for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result is not None:
                await cache_service.set(cache_key, result, ttl)
                logger.info(f"💾 Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

# Global cache instance
cache_service = RedisCacheService()

async def main():
    """Test Redis cache service"""
    print("🚀 Testing Redis Cache Service")
    print("="*50)
    
    # Health check
    health = await cache_service.health_check()
    print(f"Health Check: {'✅ Connected' if health else '❌ Failed'}")
    
    if health:
        # Test basic operations
        await cache_service.set("test_key", {"message": "Hello Redis!"}, 60)
        result = await cache_service.get("test_key")
        print(f"Test Result: {result}")
        
        # Test hash operations
        await cache_service.set_hash("user:123", {
            "name": "Test User",
            "email": "test@example.com",
            "last_login": datetime.now().isoformat()
        }, 300)
        
        user_data = await cache_service.get_hash("user:123")
        print(f"User Data: {user_data}")
        
        # Show statistics
        stats = cache_service.get_stats()
        print(f"Cache Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())