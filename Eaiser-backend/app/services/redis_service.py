#!/usr/bin/env python3
"""
Redis Caching Service for SnapFix API

This service provides high-performance caching capabilities to reduce database load
and improve API response times, especially for frequently accessed data like issues list.

Features:
- Automatic cache invalidation
- Configurable TTL (Time To Live)
- JSON serialization/deserialization
- Connection pooling
- Error handling with fallback to database
"""

import json
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class RedisService:
    """
    High-performance Redis caching service with automatic failover.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.is_connected = False
        
        # Cache configuration
        self.default_ttl = 300  # 5 minutes default TTL
        self.issues_cache_ttl = 180  # 3 minutes for issues list
        self.health_cache_ttl = 60   # 1 minute for health checks
        
        # Redis configuration from environment
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_password = os.getenv('REDIS_PASSWORD', None)
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        
    async def connect(self) -> bool:
        """
        Establish connection to Redis server with connection pooling.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create connection pool for better performance
            self.connection_pool = ConnectionPool(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            logger.info(f"✅ Redis connected successfully to {self.redis_host}:{self.redis_port}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {str(e)}. Falling back to database only.")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """
        Gracefully close Redis connection.
        """
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            self.is_connected = False
            logger.info("🔌 Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis connection: {str(e)}")
    
    def _serialize_data(self, data: Any) -> str:
        """
        Serialize data to JSON string for Redis storage.
        
        Args:
            data: Data to serialize
            
        Returns:
            str: JSON string
        """
        try:
            # Handle datetime objects
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            return json.dumps(data, default=json_serializer, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Serialization error: {str(e)}")
            raise
    
    def _deserialize_data(self, data: str) -> Any:
        """
        Deserialize JSON string from Redis.
        
        Args:
            data: JSON string from Redis
            
        Returns:
            Any: Deserialized data
        """
        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"❌ Deserialization error: {str(e)}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get data from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached data or None if not found/error
        """
        if not self.is_connected or not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(key)
            if cached_data:
                logger.debug(f"🎯 Cache HIT for key: {key}")
                return self._deserialize_data(cached_data)
            else:
                logger.debug(f"❌ Cache MISS for key: {key}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Redis GET error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Set data in Redis cache with TTL.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected or not self.redis_client:
            return False
        
        try:
            serialized_data = self._serialize_data(data)
            ttl = ttl or self.default_ttl
            
            await self.redis_client.setex(key, ttl, serialized_data)
            logger.debug(f"💾 Cache SET for key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis SET error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete data from Redis cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            logger.debug(f"🗑️ Cache DELETE for key: {key}")
            return bool(result)
        except Exception as e:
            logger.warning(f"⚠️ Redis DELETE error for key {key}: {str(e)}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "issues:*")
            
        Returns:
            int: Number of keys deleted
        """
        if not self.is_connected or not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"🧹 Cache invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"⚠️ Redis pattern invalidation error for {pattern}: {str(e)}")
            return 0
    
    # Specialized cache methods for SnapFix API
    
    async def cache_issues_list(self, issues: List[Dict], limit: int, skip: int) -> bool:
        """
        Cache issues list with pagination parameters.
        
        Args:
            issues: List of issues to cache
            limit: Pagination limit
            skip: Pagination skip
            
        Returns:
            bool: True if cached successfully
        """
        cache_key = f"issues:list:{limit}:{skip}"
        return await self.set(cache_key, issues, self.issues_cache_ttl)
    
    async def get_cached_issues_list(self, limit: int, skip: int) -> Optional[List[Dict]]:
        """
        Get cached issues list with pagination parameters.
        
        Args:
            limit: Pagination limit
            skip: Pagination skip
            
        Returns:
            Optional[List[Dict]]: Cached issues or None
        """
        cache_key = f"issues:list:{limit}:{skip}"
        return await self.get(cache_key)
    
    async def invalidate_issues_cache(self) -> int:
        """
        Invalidate all issues-related cache entries.
        
        Returns:
            int: Number of cache entries invalidated
        """
        return await self.invalidate_pattern("issues:*")
    
    async def cache_health_status(self, status: Dict) -> bool:
        """
        Cache health check status.
        
        Args:
            status: Health status data
            
        Returns:
            bool: True if cached successfully
        """
        return await self.set("health:status", status, self.health_cache_ttl)
    
    async def get_cached_health_status(self) -> Optional[Dict]:
        """
        Get cached health check status.
        
        Returns:
            Optional[Dict]: Cached health status or None
        """
        return await self.get("health:status")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        if not self.is_connected or not self.redis_client:
            return {"status": "disconnected", "keys": 0}
        
        try:
            info = await self.redis_client.info()
            keys_count = await self.redis_client.dbsize()
            
            return {
                "status": "connected",
                "keys": keys_count,
                "memory_used": info.get('used_memory_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {str(e)}")
            return {"status": "error", "error": str(e)}

# Global Redis service instance
redis_service = RedisService()

# Convenience functions for easy import
async def get_redis_service() -> RedisService:
    """
    Get the global Redis service instance.
    
    Returns:
        RedisService: Global Redis service instance
    """
    if not redis_service.is_connected:
        await redis_service.connect()
    return redis_service

async def init_redis():
    """
    Initialize Redis connection on application startup.
    """
    await redis_service.connect()

async def close_redis():
    """
    Close Redis connection on application shutdown.
    """
    await redis_service.disconnect()