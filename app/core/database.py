# 🗄️ Database Connection Utilities
# MongoDB and Redis connection management with production-ready configurations

import asyncio
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Import production-ready services
from services.mongodb_service import get_db, initialize_db
from services.redis_service import RedisService

logger = logging.getLogger(__name__)

# Global connection instances
_mongodb_client: Optional[AsyncIOMotorClient] = None
_redis_service: Optional[RedisService] = None
_redis_client: Optional[redis.Redis] = None

async def get_database():
    """
    Get MongoDB database connection using production-ready service
    Returns the database instance for queries
    """
    try:
        # Use the production-ready MongoDB service
        db = await get_db()
        if db is not None:
            logger.info("✅ MongoDB connection established via service")
            return db
        else:
            logger.warning("⚠️ MongoDB service returned None - check configuration")
            return None
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return None

async def get_redis():
    """
    Get Redis connection for caching using production-ready service
    Returns Redis client instance with graceful fallback
    """
    global _redis_service, _redis_client
    
    if _redis_service is None:
        try:
            # Initialize Redis service with production configuration
            _redis_service = RedisService()
            connection_success = await _redis_service.connect()
            
            if connection_success and _redis_service.is_connected:
                _redis_client = _redis_service.redis_client
                logger.info("✅ Redis connection established via service")
                return _redis_client
            else:
                logger.warning("⚠️ Redis service connection failed - continuing without cache")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Redis service initialization failed: {e}")
            return None
    
    # Return existing connection if available
    if _redis_service and _redis_service.is_connected:
        return _redis_service.redis_client
    else:
        return None

async def close_database_connections():
    """
    Close all database connections gracefully
    """
    global _redis_service, _redis_client
    
    # Close MongoDB connections (handled by mongodb_service)
    try:
        # MongoDB connections are handled by the service itself
        logger.info("✅ MongoDB connection will be closed by service")
    except Exception as e:
        logger.warning(f"⚠️ Error closing MongoDB: {e}")
    
    # Close Redis connections
    if _redis_service:
        try:
            await _redis_service.disconnect()
            _redis_service = None
            _redis_client = None
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing Redis: {e}")

# Dependency functions for FastAPI
async def get_db_dependency():
    """FastAPI dependency for MongoDB with production configuration"""
    return await get_database()

async def get_redis_dependency():
    """FastAPI dependency for Redis with production configuration and graceful fallback"""
    return await get_redis()