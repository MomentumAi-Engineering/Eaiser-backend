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
from pymongo.read_preferences import ReadPreference

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
        self.db_name = os.getenv("MONGODB_NAME", "eaiser_db_user")
        
        # IMPORTANT: Always use MONGODB_NAME env variable, don't extract from URI
        # This ensures consistency across all services
        logger.info(f"🔧 MongoDB Configuration:")
        logger.info(f"   URI: {self.mongo_uri[:50]}...")
        logger.info(f"   Database: {self.db_name}")
        logger.info(f"   Environment: {'Production (Atlas)' if 'mongodb+srv' in self.mongo_uri else 'Local'}")
        
        
        # Connection instances
        self.primary_client: Optional[AsyncIOMotorClient] = None
        self.read_client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.read_db: Optional[AsyncIOMotorDatabase] = None
        self.fs: Optional[motor.motor_asyncio.AsyncIOMotorGridFSBucket] = None
        
        # Circuit breakers
        self.write_circuit_breaker = MongoDBCircuitBreaker()
        self.read_circuit_breaker = MongoDBCircuitBreaker()
        
        # Performance metrics
        self.query_count = 0
        self.slow_query_count = 0
        self.error_count = 0
        self.total_query_time = 0
        
        # Connection lock
        self._connect_lock = asyncio.Lock()
        
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
                IndexModel([('status', ASCENDING), ('timestamp', DESCENDING)], name='status_timestamp'),
                IndexModel([('zip_code', ASCENDING), ('status', ASCENDING)], name='zip_code_status'),
                IndexModel([('latitude', ASCENDING), ('longitude', ASCENDING)], name='lat_lon'),
                IndexModel([('issue_type', ASCENDING), ('severity', ASCENDING)], name='issue_type_severity'),
                IndexModel([('user_email', ASCENDING)], name='user_email'),
                IndexModel([('timestamp', DESCENDING)], name='timestamp_desc'),
                IndexModel([('report_id', ASCENDING)], name='report_id'),
                IndexModel([('category', ASCENDING), ('priority', ASCENDING)], name='category_priority'),
                IndexModel([('address', TEXT), ('issue_type', TEXT)], name='address_issue_type_text'),
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
        async with self._connect_lock:
            # Check if already connected
            if self.db is not None:
                return True

            try:
                # Close existing clients if any (cleanup)
                if self.primary_client:
                    self.primary_client.close()
                if self.read_client:
                    self.read_client.close()

                # Primary client for writes (with safe options)
                self.primary_client = AsyncIOMotorClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    maxPoolSize=20,                  # Reduced for stability
                    minPoolSize=0,                   # Reduced to 0 to avoid startup hang
                    maxIdleTimeMS=30000,
                    waitQueueTimeoutMS=10000,
                    retryWrites=True,
                    compressors=['zlib'],
                )
                
                # Secondary client for reads (prefer secondary)
                self.read_client = AsyncIOMotorClient(
                    self.mongo_uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    maxPoolSize=20,                  # Reduced for stability
                    minPoolSize=0,                   # Reduced to 0 to avoid startup hang
                    maxIdleTimeMS=45000,
                    waitQueueTimeoutMS=10000,
                    compressors=['zlib'],
                    read_preference=ReadPreference.SECONDARY_PREFERRED,
                )
                
                # Test connections
                await self.primary_client.admin.command('ping')
                await self.read_client.admin.command('ping')
                
                # Initialize databases
                self.db = self.primary_client[self.db_name]
                self.read_db = self.read_client[self.db_name]
                
                # Initialize GridFS
                self.fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(self.db)
                
                # Create indexes for optimal performance
                await self._create_indexes()
                
                logger.info(f"✅ MongoDB optimized connections established")
                logger.info(f"📊 Database: {self.db_name}")
                logger.info(f"🔧 Write pool: maxPoolSize=200, Read pool: maxPoolSize=300")
                logger.info(f"⚡ Read/Write splitting enabled for 1 lakh+ users")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
                # Ensure we don't leave half-open clients
                if self.primary_client:
                    self.primary_client.close()
                    self.primary_client = None
                if self.read_client:
                    self.read_client.close()
                    self.read_client = None
                self.db = None
                self.read_db = None
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
                
                try:
                    # Create indexes (background option removed as it's deprecated/unsupported in newer Mongo versions)
                    await collection.create_indexes(indexes)
                    logger.info(f"📊 Created {len(indexes)} indexes for {collection_name} collection")
                except Exception as e:
                    # Handle IndexOptionsConflict (Code 85) - Index exists with different name/options
                    if "IndexOptionsConflict" in str(e) or "already exists" in str(e) or getattr(e, "code", 0) == 85:
                         logger.info(f"ℹ️ indexes for '{collection_name}' already exist (skipping update).")
                    else:
                        logger.error(f"❌ Failed to create indexes for {collection_name}: {str(e)}")
            
            logger.info("✅ All database indexes created/verified")
            
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
        
        if db is None:
            # Attempt to reconnect safely
            logger.warning("MongoDB DB handle is None; attempting reconnect...")
            if await self.connect():
                db = self.read_db if read_only else self.db
            
            if db is None:
                raise ConnectionFailure("MongoDB database not initialized")
                
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
        # NOTE: Add defensive defaults to avoid unbounded/slow queries on miss
        default_limit = 50 if (limit is None or (isinstance(limit, int) and limit <= 0)) else limit
        safe_skip = max(0, skip or 0)
        logger.info(f"❌ Cache MISS: Fetching from MongoDB (limit: {default_limit}, skip: {safe_skip})")
        
        # Read per-query list timeout (ms) from environment for client-side guard
        # Comment: Prevents lingering waits even if server-side max_time_ms fails to abort
        LIST_TIMEOUT_MS = int(os.getenv("LIST_TIMEOUT_MS", "5000"))
        LIST_TIMEOUT_S = max(1.0, LIST_TIMEOUT_MS / 1000.0)
        
        async with self._safe_operation('read'):
            collection = await self.get_collection(collection_name, read_only=True)
            
            # Build query cursor
            # - Add per-query timeout and batch size to prevent hanging
            # - Apply sort/pagination safely
            cursor = collection.find(filter_dict)

            # Apply defensive per-query options
            cursor = cursor.max_time_ms(5000)  # Abort if server takes >5s
            cursor = cursor.batch_size(500)    # Reasonable batch to stream results

            if sort:
                cursor = cursor.sort(sort)
            if safe_skip > 0:
                cursor = cursor.skip(safe_skip)
            if default_limit and default_limit > 0:
                cursor = cursor.limit(default_limit)

            # Execute query and convert to list
            # Always use a concrete length to avoid unbounded cursor consumption
            try:
                # Cooperative client-side timeout to complement server-side max_time_ms
                results = await asyncio.wait_for(cursor.to_list(length=default_limit), timeout=LIST_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ MongoDB find_with_cache timed out after {LIST_TIMEOUT_MS} ms (limit={default_limit}, skip={safe_skip})")
                results = []
            except Exception as e:
                # Catch timeouts or server-side query failures and fail fast
                logger.error(f"❌ MongoDB find_with_cache failed: {str(e)}")
                results = []
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            # Cache the results
            if redis_service and cache_key:
                # Cache only if we have a result set (including empty arrays for consistency)
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
            # - Add per-query timeout and batch size to prevent hanging on large pipelines
            cursor = collection.aggregate(pipeline, allowDiskUse=True)
            cursor = cursor.max_time_ms(5000).batch_size(500)
            
            # Keep length=None to respect pipeline limits; timeout still applies
            try:
                results = await cursor.to_list(length=None)
            except Exception as e:
                # Fail fast on aggregation timeout/errors to avoid hanging requests
                logger.error(f"❌ MongoDB aggregate_with_cache failed: {str(e)}")
                results = []
            
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
    
    async def get_issue_by_id(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single issue by ID with optimized query.
        
        Args:
            issue_id: The issue ID to fetch
            
        Returns:
            Optional[Dict]: Issue data or None if not found
        """
        try:
            async with self._safe_operation('read'):
                collection = await self.get_collection('issues', read_only=True)
                
                # Query by ID with per-query maxTimeMS to avoid long waits
                # Motor supports max_time_ms in find_one kwargs
                issue_data = await collection.find_one({"_id": issue_id}, max_time_ms=3000)
                
                # Fallback to ObjectId if string search fails (handles legacy/mixed data)
                if not issue_data:
                    try:
                        from bson.objectid import ObjectId
                        issue_data = await collection.find_one({"_id": ObjectId(issue_id)}, max_time_ms=3000)
                    except Exception:
                        pass
                
                if issue_data:
                    # Convert ObjectId to string for JSON serialization
                    if '_id' in issue_data:
                        issue_data['_id'] = str(issue_data['_id'])
                    
                    logger.info(f"✅ Found issue {issue_id}")
                    return issue_data
                else:
                    logger.warning(f"❌ Issue {issue_id} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to get issue {issue_id}: {str(e)}")
            raise e
    
    async def get_issues_optimized(self, filter_query: Dict[str, Any], skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get multiple issues with optimized query and pagination.
        
        Args:
            filter_query: MongoDB filter query
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List[Dict]: List of issue data
        """
        try:
            async with self._safe_operation('read'):
                collection = await self.get_collection('issues', read_only=True)
                
                # Read per-query list timeout (ms) from environment for client-side guard
                # Comment: Keeps listing responsive under intermittent network lag
                LIST_TIMEOUT_MS = int(os.getenv("LIST_TIMEOUT_MS", "5000"))
                LIST_TIMEOUT_S = max(1.0, LIST_TIMEOUT_MS / 1000.0)

                # Build optimized query cursor
                cursor = collection.find(filter_query)
                cursor = cursor.max_time_ms(5000).batch_size(500)  # Defensive query options
                
                # Apply sorting by timestamp (newest first)
                cursor = cursor.sort([('timestamp', DESCENDING)])
                
                # Apply pagination
                if skip > 0:
                    cursor = cursor.skip(skip)
                if limit > 0:
                    cursor = cursor.limit(limit)
                
                # Execute query and convert to list with cooperative timeout
                try:
                    issues = await asyncio.wait_for(cursor.to_list(length=limit), timeout=LIST_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ MongoDB get_issues_optimized timed out after {LIST_TIMEOUT_MS} ms (limit={limit}, skip={skip}, filter={filter_query})")
                    issues = []

                # Convert ObjectId to string for JSON serialization
                for issue in issues:
                    if '_id' in issue:
                        issue['_id'] = str(issue['_id'])
                
                logger.info(f"✅ Found {len(issues)} issues with filter {filter_query}")
                return issues
                
        except Exception as e:
            logger.error(f"❌ Failed to get issues with filter {filter_query}: {str(e)}")
            raise e
    
    async def update_issue_status(self, issue_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update issue status with optimized operation.
        
        Args:
            issue_id: The issue ID to update
            update_data: Data to update
            
        Returns:
            bool: True if update was successful
        """
        try:
            async with self._safe_operation('write'):
                collection = await self.get_collection('issues', read_only=False)
                
                # Update the issue
                result = await collection.update_one(
                    {"_id": issue_id},
                    {"$set": update_data}
                )
                
                success = result.modified_count > 0
                
                if success:
                    logger.info(f"✅ Updated issue {issue_id} status")
                else:
                    logger.warning(f"❌ No changes made to issue {issue_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"❌ Failed to update issue {issue_id}: {str(e)}")
            raise e
    
    async def soft_delete_issue(self, issue_id: str) -> bool:
        """
        Soft delete an issue by marking it as deleted.
        
        Args:
            issue_id: The issue ID to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Use the existing update method to mark as deleted
            update_data = {
                "status": "deleted",
                "deleted_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            return await self.update_issue_status(issue_id, update_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to soft delete issue {issue_id}: {str(e)}")
            raise e
    
    async def store_issue_optimized(self, issue_doc: Dict[str, Any], image_content: bytes) -> str:
        """
        Store issue with optimized batch operation.
        
        Args:
            issue_doc: Issue document to store
            image_content: Image binary content
            
        Returns:
            str: Inserted document ID
        """
        try:
            # Store the main issue document
            issue_id = await self.insert_one_optimized('issues', issue_doc)
            
            # Store image in GridFS if provided
            if image_content and self.fs:
                try:
                    # Optimized image storage using GridFS
                    image_id = await self.fs.upload_from_stream(
                        filename=f"{issue_doc['_id']}.jpg",
                        source=image_content,
                        metadata={"issue_id": issue_doc['_id']}
                    )
                    
                    # Update issue document with image_id
                    await self.update_one_optimized(
                        'issues',
                        {"_id": issue_doc['_id']},
                        {"$set": {
                            "image_id": str(image_id),
                            "image_stored": True, 
                            "image_size": len(image_content)
                        }}
                    )
                    logger.info(f"📸 Image stored in GridFS for issue {issue_doc['_id']} with ID {image_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store image for issue {issue_doc['_id']}: {e}")
            
            logger.info(f"✅ Stored issue {issue_doc['_id']} with optimized operation")
            return issue_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store issue: {str(e)}")
            raise e
    
    async def get_issue_image_stream(self, issue_id: str):
        """
        Retrieve image content stream from GridFS by issue ID.
        """
        try:
            async with self._safe_operation('read'):
                collection = await self.get_collection('issues', read_only=True)
                issue = await collection.find_one({"_id": issue_id})
                
                if not issue or not issue.get('image_id'):
                    return None
                
                if not self.fs:
                    return None
                    
                image_id = issue['image_id']
                try:
                    gridout = await self.fs.open_download_stream(ObjectId(image_id))
                    return gridout
                except Exception as e:
                    logger.error(f"❌ Failed to open download stream for image {image_id}: {e}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error retrieving image stream for issue {issue_id}: {e}")
            return None
    
    async def get_analytics_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get analytics summary for the specified date range.
        
        Args:
            start_date: Start date for analytics
            end_date: End date for analytics
            
        Returns:
            Dict: Analytics summary data
        """
        try:
            async with self._safe_operation('read'):
                collection = await self.get_collection('issues', read_only=True)
                
                # Build aggregation pipeline
                pipeline = [
                    {
                        "$match": {
                            "timestamp": {
                                "$gte": start_date,
                                "$lte": end_date
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "total_issues": {"$sum": 1},
                            "open_issues": {
                                "$sum": {
                                    "$cond": [{"$in": ["$status", ["pending", "needs_review", "submitted", "under_review"]]}, 1, 0]
                                }
                            },
                            "resolved_issues": {
                                "$sum": {
                                    "$cond": [{"$in": ["$status", ["resolved", "completed", "accepted", "rejected", "declined"]]}, 1, 0]
                                }
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "total_issues": 1,
                            "open_issues": 1,
                            "resolved_issues": 1
                        }
                    }
                ]
                
                # Execute aggregation
                cursor = collection.aggregate(pipeline)
                result = await cursor.to_list(length=1)
                
                analytics_data = result[0] if result else {
                    "total_issues": 0,
                    "open_issues": 0,
                    "resolved_issues": 0
                }
                
                # Add issues by type and severity
                type_pipeline = [
                    {
                        "$match": {
                            "timestamp": {
                                "$gte": start_date,
                                "$lte": end_date
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": "$issue_type",
                            "count": {"$sum": 1}
                        }
                    }
                ]
                
                severity_pipeline = [
                    {
                        "$match": {
                            "timestamp": {
                                "$gte": start_date,
                                "$lte": end_date
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": "$severity",
                            "count": {"$sum": 1}
                        }
                    }
                ]
                
                type_cursor = collection.aggregate(type_pipeline)
                severity_cursor = collection.aggregate(severity_pipeline)
                
                type_results = await type_cursor.to_list(length=None)
                severity_results = await severity_cursor.to_list(length=None)
                
                analytics_data["issues_by_type"] = {
                    item["_id"]: item["count"] for item in type_results
                }
                
                analytics_data["issues_by_severity"] = {
                    item["_id"]: item["count"] for item in severity_results
                }
                
                # Add average resolution time (placeholder for now)
                analytics_data["average_resolution_time"] = 0.0
                
                logger.info(f"✅ Generated analytics summary for date range {start_date} to {end_date}")
                return analytics_data
                
        except Exception as e:
            logger.error(f"❌ Failed to generate analytics summary: {str(e)}")
            raise e

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
            # Guard against uninitialized clients
            if not self.primary_client or not self.read_client:
                raise Exception('MongoDB clients not initialized')
            
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
