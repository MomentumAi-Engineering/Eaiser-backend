# 🗄️ Database Connection Utilities
# MongoDB and Redis connection management

import asyncio
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)

# Global connection instances
_mongodb_client: Optional[AsyncIOMotorClient] = None
_redis_client: Optional[redis.Redis] = None

async def get_database():
    """
    Get MongoDB database connection
    Returns the database instance for queries
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        try:
            # Connect to MongoDB (using existing connection from services)
            _mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017/")
            # Test connection
            await _mongodb_client.admin.command('ping')
            logger.info("✅ MongoDB connection established")
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            # Return None for graceful degradation
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected MongoDB error: {e}")
            return None
    
    return _mongodb_client.eaiser_db

async def get_redis():
    """
    Get Redis connection for caching
    Returns Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            # Connect to Redis
            _redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await _redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            # Return None for graceful degradation
            return None
    
    return _redis_client

async def close_database_connections():
    """
    Close all database connections
    """
    global _mongodb_client, _redis_client
    
    if _mongodb_client:
        _mongodb_client.close()
        _mongodb_client = None
        logger.info("✅ MongoDB connection closed")
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("✅ Redis connection closed")

# Dependency functions for FastAPI
async def get_db_dependency():
    """FastAPI dependency for MongoDB"""
    return await get_database()

async def get_redis_dependency():
    """FastAPI dependency for Redis"""
    return await get_redis()