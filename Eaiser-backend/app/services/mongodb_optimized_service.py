#!/usr/bin/env python3
"""
Optimized MongoDB Service for High-Load Operations (1 Lakh+ Users)

This service provides enterprise-grade MongoDB operations with advanced
optimizations for handling massive concurrent traffic, including:

- Connection pooling and load balancing
- Query optimization and indexing
- Read/Write splitting
- Automatic retry and circuit breaker
- Performance monitoring and metrics
- Batch operations for efficiency
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, ConnectionFailure, ServerSelectionTimeoutError
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
import time
from contextlib import asynccontextmanager
from services.redis_cluster_service import get_redis_cluster_service

load_dotenv()
logger = logging.getLogger(__name__)

class MongoDBCircuitBreaker:
    """Circuit breaker for MongoDB operations"""
    
    def __init__(self, failure_threshold=10, recovery_timeout=60):
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

class OptimizedMongoDBService:
    """
    Enterprise MongoDB service optimized for 1 lakh+ concurrent users.
    """
    
    def __init__(self):
        # Connection configuration
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("MONGODB_NAME", "snapfix")
        
        # Parse database name from URI if provided
        parsed_uri = urlparse(self.mongo_uri)
        if parsed_uri.path and parsed_uri.path.strip("/"):
            self.db_name = parsed_uri.path.strip("/")
        
        # Connection instances
        self.primary_client: Optional[AsyncIOMotorClient] = None
        self.read_client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.read_db: Optional[AsyncIOMotorDatabase] = None
        
        # Circuit breakers
        self.write_circuit_breaker = MongoDBCircuitBreaker()
        self.read_circuit_breaker = MongoDBCircuitBreaker()
        
        # Performance metrics
        self.query_count = 0
        self.slow_query_count = 0
        self.error_count = 0
        self.total_query_time = 0
        
        # Collection names
        self.collections = {
            'issues': 'issues',
            'users': 'users',
            'authorities': 'authorities',
            'reports': 'reports',
            'analytics': 'analytics'
        }
        
        # Index definitions for optimal query performance
        self.index_definitions = {
            'issues': [
                IndexModel([('status', ASCENDING), ('timestamp', DESCENDING)]),
                IndexModel([('zip_code', ASCENDING), ('status', ASCENDING)]),
                IndexModel([('latitude', ASCENDING), ('longitude', ASCENDING)]),
                IndexModel([('issue_type', ASCENDING), ('severity', ASCENDING)]),
                IndexModel([('user_email', ASCENDING)]),
                IndexModel([('timestamp', DESCENDING)]),
                IndexModel([('report_id', ASCENDING)]),
                IndexModel([('category', ASCENDING), ('priority', ASCENDING)]),
                IndexModel([('address', TEXT), ('issue_type', TEXT)]),  # Text search
            ],
            'users': [
                IndexModel([('email', ASCENDING)], unique=True),
                IndexModel([('created_at', DESCENDING)]),
                IndexModel([('last_login', DESCENDING)]),
            ],
            'authorities': [
                IndexModel([('zip_code', ASCENDING), ('type', ASCENDING)]),
                IndexModel([('email', ASCENDING)]),
                IndexModel([('name', TEXT)]),
            ],
            'reports': [
                IndexModel([('issue_id', ASCENDING)]),
                IndexModel([('created_at', DESCENDING)]),
                IndexModel([('status', ASCENDING)]),
            ]
        }
    
    async def connect(self) -> bool:
        """
        Establish optimized MongoDB connections with read/write splitting.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Primary client for writes (with write concern)
            self.primary_client = AsyncIOMotorClient(
                self.mongo_uri,
                # Connection timeout settings
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                
                # Connection pooling for high performance
                maxPoolSize=200,        # Increased for 1 lakh users
                minPoolSize=20,         # Minimum connections
                maxIdleTimeMS=30000,    # Keep connections alive
                waitQueueTimeoutMS=5000,
                
                # Write optimizations
                retryWrites=True,
                w="majority",           # Write concern for consistency
                j=True,                 # Journal writes
                
                # Performance optimizations
                compressors="snappy,zlib",
                zlibCompressionLevel=6,
                
                # Read preference for writes
                readPreference="primary"
            )
            
            # Secondary client for reads (with read preference)
            self.read_client = AsyncIOMotorClient(
                self.mongo_uri,
                # Connection settings
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=8000,
                socketTimeoutMS=15000,
                
                # Connection pooling for reads
                maxPoolSize=300,        # More connections for reads
                minPoolSize=30,
                maxIdleTimeMS=45000,
                waitQueueTimeoutMS=3000,
                
                # Read optimizations
                readPreference="secondaryPreferred",  # Prefer secondary for reads
                readConcern={"level": "majority"},
                
                # Performance optimizations
                compressors="snappy,zlib",
                zlibCompressionLevel=6
            )
            
            # Test connections
            await self.primary_client.admin.command('ping')
            await self.read_client.admin.command('ping')
            
            # Initialize databases
            self.db = self.primary_client[self.db_name]
            self.read_db = self.read_client[self.db_name]
            
            # Create indexes for optimal performance
            await self._create_indexes()
            
            logger.info(f"✅ MongoDB optimized connections established")
            logger.info(f"📊 Database: {self.db_name}")
            logger.info(f"🔧 Write pool: maxPoolSize=200, Read pool: maxPoolSize=300")
            logger.info(f"⚡ Read/Write splitting enabled for 1 lakh+ users")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            return False
    
    async def disconnect(self):
        """Close MongoDB connections gracefully."""
        try:
            if self.primary_client:
                self.primary_client.close()
            if self.read_client:
                self.read_client.close()
            logger.info("🔒 MongoDB connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing MongoDB connections: {str(e)}")
    
    async def _create_indexes(self):
        """Create optimized indexes for all collections."""
        try:
            for collection_name, indexes in self.index_definitions.items():
                collection = self.db[collection_name]
                
                # Create indexes in background for better performance
                await collection.create_indexes(indexes, background=True)
                
                logger.info(f"📊 Created {len(indexes)} indexes for {collection_name} collection")
            
            logger.info("✅ All database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {str(e)}")
    
    @asynccontextmanager
    async def _safe_operation(self, operation_type='read'):
        """Context manager for safe database operations with circuit breaker."""
        circuit_breaker = self.read_circuit_breaker if operation_type == 'read' else self.write_circuit_breaker
        
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker is OPEN for {operation_type} operations")
        
        start_time = time.time()
        
        try:
            yield
            circuit_breaker.record_success()
            
            # Track performance metrics
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            self.query_count += 1
            self.total_query_time += query_time
            
            if query_time > 1000:  # Slow query threshold: 1 second
                self.slow_query_count += 1
                logger.warning(f"⚠️ Slow {operation_type} query detected: {query_time:.2f}ms")
            
            logger.debug(f"✅ MongoDB {operation_type.upper()} took {query_time:.2f} ms")
            
        except Exception as e:
            circuit_breaker.record_failure()
            self.error_count += 1
            query_time = (time.time() - start_time) * 1000
            logger.error(f"❌ MongoDB {operation_type} error after {query_time:.2f}ms: {str(e)}")
            raise e
    
    async def get_collection(self, collection_name: str, read_only: bool = True) -> AsyncIOMotorCollection:
        """
        Get collection with appropriate read/write client.
        
        Args:
            collection_name: Name of the collection
            read_only: Whether this is a read-only operation
        
        Returns:
            AsyncIOMotorCollection: Collection instance
        """
        db = self.read_db if read_only else self.db
        return db[self.collections.get(collection_name, collection_name)]
    
    async def find_with_cache(self, collection_name: str, filter_dict: Dict, 
                             cache_key: str = None, cache_ttl: int = 300,
                             limit: int = None, skip: int = None, 
                             sort: List[tuple] = None) -> List[Dict]:
        """
        Find documents with intelligent caching.
        
        Args:
            collection_name: Collection to query
            filter_dict: MongoDB filter
            cache_key: Custom cache key
            cache_ttl: Cache TTL in seconds
            limit: Limit results
            skip: Skip results
            sort: Sort specification
        
        Returns:
            List[Dict]: Query results
        """
        # Try cache first
        redis_service = await get_redis_cluster_service()
        
        if redis_service and cache_key:
            cached_result = await redis_service.get_cache('api_response', cache_key)
            if cached_result is not None:
                logger.info(f"✅ Cache HIT: Fetching from Redis")
                return cached_result
        
        # Cache miss - query database
        logger.info(f"❌ Cache MISS: Fetching from MongoDB (limit: {limit}, skip: {skip})")
        
        async with self._safe_operation('read'):
            collection = await self.get_collection(collection_name, read_only=True)
            
            # Build query cursor
            cursor = collection.find(filter_dict)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            # Execute query and convert to list
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            # Cache the results
            if redis_service and cache_key:
                await redis_service.set_cache('api_response', cache_key, results, cache_ttl)
            
            return results
    
    async def aggregate_with_cache(self, collection_name: str, pipeline: List[Dict],
                                  cache_key: str = None, cache_ttl: int = 300) -> List[Dict]:
        """
        Execute aggregation pipeline with caching.
        
        Args:
            collection_name: Collection to aggregate
            pipeline: Aggregation pipeline
            cache_key: Custom cache key
            cache_ttl: Cache TTL in seconds
        
        Returns:
            List[Dict]: Aggregation results
        """
        # Try cache first
        redis_service = await get_redis_cluster_service()
        
        if redis_service and cache_key:
            cached_result = await redis_service.get_cache('api_response', cache_key)
            if cached_result is not None:
                logger.info(f"✅ Cache HIT: Aggregation from Redis")
                return cached_result
        
        # Cache miss - execute aggregation
        logger.info(f"❌ Cache MISS: Executing aggregation pipeline")
        
        async with self._safe_operation('read'):
            collection = await self.get_collection(collection_name, read_only=True)
            
            # Execute aggregation
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            # Cache the results
            if redis_service and cache_key:
                await redis_service.set_cache('api_response', cache_key, results, cache_ttl)
            
            return results
    
    async def insert_one_optimized(self, collection_name: str, document: Dict) -> str:
        """
        Insert single document with optimizations.
        
        Args:
            collection_name: Collection to insert into
            document: Document to insert
        
        Returns:
            str: Inserted document ID
        """
        async with self._safe_operation('write'):
            collection = await self.get_collection(collection_name, read_only=False)
            
            # Add timestamp if not present
            if 'created_at' not in document:
                document['created_at'] = datetime.utcnow()
            
            result = await collection.insert_one(document)
            
            # Invalidate related cache
            redis_service = await get_redis_cluster_service()
            if redis_service:
                await redis_service.invalidate_pattern(f"{collection_name}:*")
            
            return str(result.inserted_id)
    
    async def insert_many_batch(self, collection_name: str, documents: List[Dict], 
                               batch_size: int = 1000) -> List[str]:
        """
        Insert multiple documents in optimized batches.
        
        Args:
            collection_name: Collection to insert into
            documents: Documents to insert
            batch_size: Batch size for insertion
        
        Returns:
            List[str]: List of inserted document IDs
        """
        inserted_ids = []
        
        # Add timestamps
        current_time = datetime.utcnow()
        for doc in documents:
            if 'created_at' not in doc:
                doc['created_at'] = current_time
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            async with self._safe_operation('write'):
                collection = await self.get_collection(collection_name, read_only=False)
                
                try:
                    result = await collection.insert_many(batch, ordered=False)
                    batch_ids = [str(id) for id in result.inserted_ids]
                    inserted_ids.extend(batch_ids)
                    
                    logger.info(f"✅ Batch insert: {len(batch)} documents into {collection_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Batch insert failed for {collection_name}: {str(e)}")
                    # Continue with next batch
                    continue
        
        # Invalidate related cache
        redis_service = await get_redis_cluster_service()
        if redis_service:
            await redis_service.invalidate_pattern(f"{collection_name}:*")
        
        return inserted_ids
    
    async def update_one_optimized(self, collection_name: str, filter_dict: Dict, 
                                  update_dict: Dict, upsert: bool = False) -> bool:
        """
        Update single document with optimizations.
        
        Args:
            collection_name: Collection to update
            filter_dict: Filter for document to update
            update_dict: Update operations
            upsert: Whether to insert if not found
        
        Returns:
            bool: True if document was modified
        """
        async with self._safe_operation('write'):
            collection = await self.get_collection(collection_name, read_only=False)
            
            # Add updated timestamp
            if '$set' not in update_dict:
                update_dict['$set'] = {}
            update_dict['$set']['updated_at'] = datetime.utcnow()
            
            result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
            
            # Invalidate related cache
            redis_service = await get_redis_cluster_service()
            if redis_service:
                await redis_service.invalidate_pattern(f"{collection_name}:*")
            
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
    
    async def delete_one_optimized(self, collection_name: str, filter_dict: Dict) -> bool:
        """
        Delete single document with cache invalidation.
        
        Args:
            collection_name: Collection to delete from
            filter_dict: Filter for document to delete
        
        Returns:
            bool: True if document was deleted
        """
        async with self._safe_operation('write'):
            collection = await self.get_collection(collection_name, read_only=False)
            
            result = await collection.delete_one(filter_dict)
            
            # Invalidate related cache
            redis_service = await get_redis_cluster_service()
            if redis_service:
                await redis_service.invalidate_pattern(f"{collection_name}:*")
            
            return result.deleted_count > 0
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database performance statistics.
        
        Returns:
            Dict: Performance metrics and statistics
        """
        stats = {
            'query_count': self.query_count,
            'slow_query_count': self.slow_query_count,
            'error_count': self.error_count,
            'average_query_time_ms': (self.total_query_time / max(self.query_count, 1)),
            'slow_query_percentage': (self.slow_query_count / max(self.query_count, 1)) * 100,
            'error_percentage': (self.error_count / max(self.query_count, 1)) * 100,
            'read_circuit_breaker_state': self.read_circuit_breaker.state,
            'write_circuit_breaker_state': self.write_circuit_breaker.state
        }
        
        try:
            # Get MongoDB server stats
            if self.db:
                server_status = await self.db.command('serverStatus')
                stats.update({
                    'mongodb_version': server_status.get('version', 'unknown'),
                    'uptime_seconds': server_status.get('uptime', 0),
                    'connections_current': server_status.get('connections', {}).get('current', 0),
                    'connections_available': server_status.get('connections', {}).get('available', 0),
                    'opcounters_query': server_status.get('opcounters', {}).get('query', 0),
                    'opcounters_insert': server_status.get('opcounters', {}).get('insert', 0),
                    'opcounters_update': server_status.get('opcounters', {}).get('update', 0),
                    'opcounters_delete': server_status.get('opcounters', {}).get('delete', 0)
                })
        except Exception as e:
            logger.error(f"❌ Failed to get MongoDB server stats: {str(e)}")
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dict: Health status and metrics
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        try:
            # Test primary connection
            start_time = time.time()
            await self.primary_client.admin.command('ping')
            primary_latency = (time.time() - start_time) * 1000
            
            health_status['checks']['primary_connection'] = {
                'status': 'healthy',
                'latency_ms': round(primary_latency, 2)
            }
            
            # Test read connection
            start_time = time.time()
            await self.read_client.admin.command('ping')
            read_latency = (time.time() - start_time) * 1000
            
            health_status['checks']['read_connection'] = {
                'status': 'healthy',
                'latency_ms': round(read_latency, 2)
            }
            
            # Test database operations
            start_time = time.time()
            test_collection = await self.get_collection('issues', read_only=True)
            await test_collection.count_documents({})
            query_latency = (time.time() - start_time) * 1000
            
            health_status['checks']['database_query'] = {
                'status': 'healthy',
                'latency_ms': round(query_latency, 2)
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"❌ MongoDB health check failed: {str(e)}")
        
        return health_status

# Global optimized MongoDB service instance
_optimized_mongodb_service: Optional[OptimizedMongoDBService] = None

async def init_optimized_mongodb() -> OptimizedMongoDBService:
    """
    Initialize optimized MongoDB service.
    
    Returns:
        OptimizedMongoDBService: Initialized service instance
    """
    global _optimized_mongodb_service
    
    if _optimized_mongodb_service is None:
        _optimized_mongodb_service = OptimizedMongoDBService()
        await _optimized_mongodb_service.connect()
    
    return _optimized_mongodb_service

async def get_optimized_mongodb_service() -> Optional[OptimizedMongoDBService]:
    """
    Get optimized MongoDB service instance.
    
    Returns:
        Optional[OptimizedMongoDBService]: Service instance or None if not initialized
    """
    return _optimized_mongodb_service

async def close_optimized_mongodb():
    """
    Close optimized MongoDB service.
    """
    global _optimized_mongodb_service
    
    if _optimized_mongodb_service:
        await _optimized_mongodb_service.disconnect()
        _optimized_mongodb_service = None
