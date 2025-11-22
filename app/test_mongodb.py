#!/usr/bin/env python3
"""
Test script to debug MongoDB connection and query issues
"""
import asyncio
import logging
import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mongodb_optimized_service import init_optimized_mongodb, get_optimized_mongodb_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mongodb_service():
    """Test MongoDB service functionality"""
    try:
        logger.info("🚀 Starting MongoDB service test...")
        
        # Initialize the service
        logger.info("🔧 Initializing MongoDB service...")
        await init_optimized_mongodb()
        
        # Get the service instance
        service = await get_optimized_mongodb_service()
        if not service:
            logger.error("❌ MongoDB service not available")
            return
        
        logger.info("✅ MongoDB service initialized successfully")
        
        # Test health check
        logger.info("🏥 Testing health check...")
        health = await service.health_check()
        logger.info(f"Health status: {health}")
        
        # Test getting issues
        logger.info("📋 Testing get_issues_optimized...")
        start_time = time.time()
        
        try:
            issues = await service.get_issues_optimized({}, skip=0, limit=5)
            query_time = time.time() - start_time
            
            logger.info(f"✅ Found {len(issues)} issues in {query_time:.2f} seconds")
            
            if issues:
                logger.info(f"First issue ID: {issues[0].get('_id', 'No ID')}")
                logger.info(f"First issue keys: {list(issues[0].keys())}")
            else:
                logger.info("No issues found in database")
                
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Test getting a single issue if any exist
        if issues:
            logger.info("🔍 Testing get_issue_by_id...")
            first_issue_id = issues[0].get('_id')
            if first_issue_id:
                single_issue = await service.get_issue_by_id(first_issue_id)
                if single_issue:
                    logger.info(f"✅ Successfully retrieved single issue: {single_issue.get('_id')}")
                else:
                    logger.warning("❌ Single issue not found")
        
        logger.info("✅ MongoDB service test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mongodb_service())