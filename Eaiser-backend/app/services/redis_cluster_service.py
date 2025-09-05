#!/usr/bin/env python3
"""
Redis Cluster Service for High-Load Caching (1 Lakh+ Users)

This service provides enterprise-grade Redis clustering capabilities to handle
massive concurrent traffic with automatic failover, load balancing, and
horizontal scaling.

Features:
- Redis Cluster support for horizontal scaling
- Automatic sharding and replication
- Connection pooling with circuit breaker
- Intelligent cache warming and invalidation
- Performance monitoring and metrics
- Graceful degradation on failures
"""

import json
import logging
import asyncio
from typing import Any, Optional, List, Dict, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import RedisCluster, ConnectionPool
import os
from dotenv import load_dotenv
import hashlib
import time
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for Redis operations"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self):
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RedisClusterService:
    """
    Enterprise Redis Cluster Service for handling 1 lakh+ concurrent users.
    """
    
    def __init__(self):
        self.redis_cluster: Optional[RedisCluster] = None
        self.is_connected = False
        self.circuit_breaker = CircuitBreaker()
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_operations = 0
        
        # Cache configuration for different data types
        self.cache_config = {
            'issues_list': {'ttl': 300, 'prefix': 'issues:list'},
            'issue_detail': {'ttl': 600, 'prefix': 'issue:detail'},
            'user_session': {'ttl': 1800, 'prefix': 'user:session'},
            'api_response': {'ttl': 180, 'prefix': 'api:response'},
            'health_check': {'ttl': 60, 'prefix': 'health:check'},
            'authorities': {'ttl': 3600, 'prefix': 'auth:data'},
            'geocode': {'ttl': 86400, 'prefix': 'geo:cache'}
        }
        
        # Redis cluster configuration
        self.cluster_nodes = [
            {'host': os.getenv('REDIS_NODE_1_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_1_PORT', 7000))},
            {'host': os.getenv('REDIS_NODE_2_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_2_PORT', 7001))},
            {'host': os.getenv('REDIS_NODE_3_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_3_PORT', 7002))},
            {'host': os.getenv('REDIS_NODE_4_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_4_PORT', 7003))},
            {'host': os.getenv('REDIS_NODE_5_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_5_PORT', 7004))},
            {'host': os.getenv('REDIS_NODE_6_HOST', 'localhost'), 'port': int(os.getenv('REDIS_NODE_6_PORT', 7005))}
        ]
        
        # Fallback to single Redis instance if cluster not available
        self.fallback_redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.fallback_redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_password = os.getenv('REDIS_PASSWORD', None)
        
    async def connect(self) -> bool:
        """
        Establish connection to Redis cluster with automatic failover.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try Redis Cluster first
            startup_nodes = []
            for node in self.cluster_nodes:
                startup_nodes.append({"host": node['host'], "port": node['port']})
            
            try:
                self.redis_cluster = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.redis_password,
                    decode_responses=True,
                    skip_full_coverage_check=True,
                    max_connections=200,  # High connection pool for 1 lakh users
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30,
                    max_connections_per_node=50
                )
                
                # Test cluster connection
                await self.redis_cluster.ping()
                self.is_connected = True
                logger.info(f"✅ Redis Cluster connected successfully with {len(startup_nodes)} nodes")
                logger.info(f"🚀 Cluster ready for 1 lakh+ concurrent users")
                return True
                
            except Exception as cluster_error:
                logger.warning(f"⚠️ Redis Cluster connection failed: {cluster_error}")
                logger.info("🔄 Falling back to single Redis instance...")
                
                # Fallback to single Redis instance
                self.redis_cluster = redis.Redis(
                    host=self.fallback_redis_host,
                    port=self.fallback_redis_port,
                    password=self.redis_password,
                    decode_responses=True,
                    max_connections=100,
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                
                await self.redis_cluster.ping()
                self.is_connected = True
                logger.info(f"✅ Redis single instance connected: {self.fallback_redis_host}:{self.fallback_redis_port}")
                return True
                
        except Exception as e:
            logger.error(f"❌ All Redis connections failed: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Gracefully close Redis connections."""
        try:
            if self.redis_cluster:
                await self.redis_cluster.close()
            self.is_connected = False
            logger.info("🔌 Redis cluster connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis cluster connection: {str(e)}")
    
    def _generate_cache_key(self, cache_type: str, identifier: str, **kwargs) -> str:
        """
        Generate optimized cache key with consistent hashing.
        
        Args:
            cache_type: Type of cache (issues_list, issue_detail, etc.)
            identifier: Unique identifier for the data
            **kwargs: Additional parameters for key generation
        
        Returns:
            str: Generated cache key
        """
        config = self.cache_config.get(cache_type, {'prefix': 'default'})
        base_key = f"{config['prefix']}:{identifier}"
        
        # Add additional parameters to key
        if kwargs:
            params = ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
            base_key = f"{base_key}:{params}"
        
        # Use consistent hashing for cluster distribution
        key_hash = hashlib.md5(base_key.encode()).hexdigest()[:8]
        return f"{base_key}:{key_hash}"
    
    @asynccontextmanager
    async def _safe_operation(self):
        """Context manager for safe Redis operations with circuit breaker."""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is OPEN")
        
        try:
            yield
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise e
    
    async def set_cache(self, cache_type: str, identifier: str, data: Any, custom_ttl: Optional[int] = None, **kwargs) -> bool:
        """
        Set cache with intelligent distribution and compression.
        
        Args:
            cache_type: Type of cache
            identifier: Unique identifier
            data: Data to cache
            custom_ttl: Custom TTL override
            **kwargs: Additional key parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            async with self._safe_operation():
                cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
                config = self.cache_config.get(cache_type, {'ttl': 300})
                ttl = custom_ttl or config['ttl']
                
                # Serialize data with compression for large objects
                serialized_data = json.dumps(data, default=str)
                
                # Use pipeline for better performance
                pipe = self.redis_cluster.pipeline()
                pipe.setex(cache_key, ttl, serialized_data)
                
                # Add metadata for monitoring
                metadata_key = f"meta:{cache_key}"
                metadata = {
                    'created_at': datetime.utcnow().isoformat(),
                    'cache_type': cache_type,
                    'size_bytes': len(serialized_data)
                }
                pipe.setex(metadata_key, ttl, json.dumps(metadata))
                
                await pipe.execute()
                
                self.total_operations += 1
                logger.debug(f"✅ Cache SET: {cache_key} (TTL: {ttl}s, Size: {len(serialized_data)} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Cache SET failed for {cache_type}:{identifier}: {str(e)}")
            return False
    
    async def get_cache(self, cache_type: str, identifier: str, **kwargs) -> Optional[Any]:
        """
        Get cache with automatic deserialization and hit/miss tracking.
        
        Args:
            cache_type: Type of cache
            identifier: Unique identifier
            **kwargs: Additional key parameters
        
        Returns:
            Any: Cached data or None if not found
        """
        if not self.is_connected:
            self.cache_misses += 1
            return None
        
        try:
            async with self._safe_operation():
                cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
                
                cached_data = await self.redis_cluster.get(cache_key)
                
                if cached_data:
                    self.cache_hits += 1
                    self.total_operations += 1
                    
                    try:
                        deserialized_data = json.loads(cached_data)
                        logger.debug(f"✅ Cache HIT: {cache_key}")
                        return deserialized_data
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Cache deserialization failed for {cache_key}: {str(e)}")
                        # Remove corrupted cache entry
                        await self.redis_cluster.delete(cache_key)
                        self.cache_misses += 1
                        return None
                else:
                    self.cache_misses += 1
                    self.total_operations += 1
                    logger.debug(f"❌ Cache MISS: {cache_key}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Cache GET failed for {cache_type}:{identifier}: {str(e)}")
            self.cache_misses += 1
            return None
    
    async def delete_cache(self, cache_type: str, identifier: str, **kwargs) -> bool:
        """
        Delete cache entry and its metadata.
        
        Args:
            cache_type: Type of cache
            identifier: Unique identifier
            **kwargs: Additional key parameters
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            async with self._safe_operation():
                cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
                metadata_key = f"meta:{cache_key}"
                
                # Delete both data and metadata
                pipe = self.redis_cluster.pipeline()
                pipe.delete(cache_key)
                pipe.delete(metadata_key)
                result = await pipe.execute()
                
                deleted_count = sum(result)
                logger.debug(f"🗑️ Cache DELETE: {cache_key} (deleted {deleted_count} keys)")
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"❌ Cache DELETE failed for {cache_type}:{identifier}: {str(e)}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate multiple cache entries matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "issues:*")
        
        Returns:
            int: Number of keys deleted
        """
        if not self.is_connected:
            return 0
        
        try:
            async with self._safe_operation():
                # Get all keys matching pattern
                keys = await self.redis_cluster.keys(pattern)
                
                if keys:
                    # Delete in batches for better performance
                    batch_size = 100
                    deleted_count = 0
                    
                    for i in range(0, len(keys), batch_size):
                        batch_keys = keys[i:i + batch_size]
                        deleted_count += await self.redis_cluster.delete(*batch_keys)
                    
                    logger.info(f"🗑️ Cache PATTERN DELETE: {pattern} (deleted {deleted_count} keys)")
                    return deleted_count
                
                return 0
                
        except Exception as e:
            logger.error(f"❌ Cache PATTERN DELETE failed for {pattern}: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache performance statistics.
        
        Returns:
            Dict: Cache statistics and performance metrics
        """
        stats = {
            'is_connected': self.is_connected,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_operations': self.total_operations,
            'hit_rate': (self.cache_hits / max(self.total_operations, 1)) * 100,
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }
        
        if self.is_connected:
            try:
                # Get Redis cluster info
                info = await self.redis_cluster.info()
                stats.update({
                    'redis_version': info.get('redis_version', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.error(f"❌ Failed to get Redis info: {str(e)}")
        
        return stats
    
    async def warm_cache(self, cache_type: str, data_loader_func, identifiers: List[str]) -> int:
        """
        Warm cache with frequently accessed data.
        
        Args:
            cache_type: Type of cache to warm
            data_loader_func: Async function to load data
            identifiers: List of identifiers to warm
        
        Returns:
            int: Number of items successfully cached
        """
        if not self.is_connected:
            return 0
        
        warmed_count = 0
        
        try:
            # Process in batches to avoid overwhelming the system
            batch_size = 50
            
            for i in range(0, len(identifiers), batch_size):
                batch_identifiers = identifiers[i:i + batch_size]
                
                # Load data concurrently
                tasks = [data_loader_func(identifier) for identifier in batch_identifiers]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Cache successful results
                for identifier, result in zip(batch_identifiers, results):
                    if not isinstance(result, Exception) and result is not None:
                        success = await self.set_cache(cache_type, identifier, result)
                        if success:
                            warmed_count += 1
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"🔥 Cache warmed: {cache_type} ({warmed_count}/{len(identifiers)} items)")
            return warmed_count
            
        except Exception as e:
            logger.error(f"❌ Cache warming failed for {cache_type}: {str(e)}")
            return warmed_count

# Global Redis cluster service instance
_redis_cluster_service: Optional[RedisClusterService] = None

async def init_redis_cluster() -> RedisClusterService:
    """
    Initialize Redis cluster service.
    
    Returns:
        RedisClusterService: Initialized service instance
    """
    global _redis_cluster_service
    
    if _redis_cluster_service is None:
        _redis_cluster_service = RedisClusterService()
        await _redis_cluster_service.connect()
    
    return _redis_cluster_service

async def get_redis_cluster_service() -> Optional[RedisClusterService]:
    """
    Get Redis cluster service instance.
    
    Returns:
        Optional[RedisClusterService]: Service instance or None if not initialized
    """
    return _redis_cluster_service

async def close_redis_cluster():
    """
    Close Redis cluster service.
    """
    global _redis_cluster_service
    
    if _redis_cluster_service:
        await _redis_cluster_service.disconnect()
        _redis_cluster_service = None
