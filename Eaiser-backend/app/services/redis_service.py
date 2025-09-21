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
- Production-ready configuration for Render deployment
"""

import json
import logging
import asyncio
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

def get_redis_config():
    """
    Get Redis configuration with production-ready settings for Render deployment.
    
    Returns:
        tuple: (host, port, password, db, ssl_required)
    """
    # Priority order for Redis configuration
    # 1. REDIS_URL (full connection string) - preferred for production
    # 2. Individual environment variables
    # 3. Localhost fallback for development
    
    redis_url = os.getenv('REDIS_URL')
    
    if redis_url:
        # Parse Redis URL format: redis://[:password@]host:port[/db]
        # or rediss://[:password@]host:port[/db] for SSL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            
            host = parsed.hostname or 'localhost'
            port = parsed.port or 6379
            password = parsed.password
            db = int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0
            ssl_required = parsed.scheme == 'rediss'
            
            logger.info(f"🔧 Redis URL configured: {parsed.scheme}://{host}:{port}/{db}")
            return host, port, password, db, ssl_required
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse REDIS_URL: {e}")
    
    # Fallback to individual environment variables
    host = os.getenv('REDIS_HOST', 'localhost')
    port = int(os.getenv('REDIS_PORT', 6379))
    password = os.getenv('REDIS_PASSWORD', None)
    db = int(os.getenv('REDIS_DB', 0))
    ssl_required = os.getenv('REDIS_SSL', 'false').lower() == 'true'
    
    # Log configuration (hide sensitive info)
    password_display = "***" if password else "None"
    logger.info(f"🔧 Redis Configuration:")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Password: {password_display}")
    logger.info(f"   Database: {db}")
    logger.info(f"   SSL: {ssl_required}")
    logger.info(f"   Environment: {'Production' if host != 'localhost' else 'Development'}")
    
    return host, port, password, db, ssl_required

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
        
        # Get Redis configuration
        self.redis_host, self.redis_port, self.redis_password, self.redis_db, self.ssl_required = get_redis_config()

    async def connect(self) -> bool:
        """
        Establish connection to Redis server with connection pooling.
        Production-ready configuration with SSL support and graceful fallback.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Check if Redis is available (localhost check for development)
            if self.redis_host == 'localhost':
                logger.warning("⚠️ Using localhost Redis - this will fail in production!")
                logger.warning("⚠️ Please set REDIS_URL environment variable for production deployment")
                logger.info("💡 For development, you can:")
                logger.info("   1. Install Redis locally: https://redis.io/download")
                logger.info("   2. Use Docker: docker run -d -p 6379:6379 redis:alpine")
                logger.info("   3. Skip Redis (app will work without caching)")
            
            # Create connection pool configuration
            pool_config = {
                'host': self.redis_host,
                'port': self.redis_port,
                'password': self.redis_password,
                'db': self.redis_db,
                'decode_responses': True,
                'max_connections': 20,
                'retry_on_timeout': True,
                'socket_connect_timeout': 10,  # Increased for production
                'socket_timeout': 10,          # Increased for production
                'health_check_interval': 30,   # Health check every 30 seconds
            }
            
            # Add SSL configuration for production Redis (like Redis Cloud, AWS ElastiCache)
            if self.ssl_required:
                pool_config.update({
                    'ssl': True,
                    'ssl_check_hostname': False,  # Often needed for cloud Redis services
                    'ssl_cert_reqs': None,        # Disable certificate verification for managed services
                })
                logger.info("🔒 SSL enabled for Redis connection")
            
            # Create connection pool for better performance
            self.connection_pool = ConnectionPool(**pool_config)
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=5.0)
            self.is_connected = True
            
            logger.info(f"✅ Redis connected successfully to {self.redis_host}:{self.redis_port}")
            logger.info(f"🚀 Redis caching enabled - performance optimized!")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Redis connection timeout to {self.redis_host}:{self.redis_port}")
            logger.warning("💡 This is normal if Redis is not available - app will work without caching")
            self.is_connected = False
            return False
        except ConnectionRefusedError:
            logger.warning(f"🚫 Redis connection refused to {self.redis_host}:{self.redis_port}")
            logger.warning("💡 Redis server not running - app will work without caching")
            self.is_connected = False
            return False
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {str(e)}")
            logger.warning("💡 App will continue without caching - performance may be slower")
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
    Initialize Redis service with graceful error handling.
    Application will continue to work even if Redis is unavailable.
    """
    try:
        success = await redis_service.connect()
        if success:
            logger.info("✅ Redis caching service initialized successfully")
        else:
            logger.warning("⚠️ Redis unavailable - continuing without caching")
    except Exception as e:
        logger.warning(f"⚠️ Redis initialization failed: {str(e)} - continuing without caching")

async def close_redis():
    """
    Gracefully close Redis connection.
    """
    try:
        await redis_service.disconnect()
    except Exception as e:
        logger.error(f"❌ Error closing Redis: {str(e)}")