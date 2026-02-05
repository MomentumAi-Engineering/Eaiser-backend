import logging
import re
from typing import Optional, List, Dict, Any
import motor.motor_asyncio
import gridfs
from bson.objectid import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import time  # Used for measuring query durations and timeouts
# Import URL parsing utilities correctly
# NOTE: Using direct imports avoids NameError when referencing urllib.parse
from urllib.parse import urlparse, parse_qs
from services.redis_service import get_redis_service

load_dotenv()
logger = logging.getLogger(__name__)

# Log Motor version for debugging
try:
    import motor
    # Use motor.version for version info, fallback to motor._version for older versions
    motor_version = getattr(motor, 'version', getattr(motor, '_version', 'unknown'))
    logger.info(f"Using Motor version: {motor_version}")
except ImportError:
    logger.error("Motor library not installed. Install with 'pip install motor'")
    raise

# ✅ Use the new TLS-fixed MongoDB configuration
def get_mongodb_config():
    """
    Get MongoDB configuration with fallback options
    Priority: MONGO_URI > MONGODB_URL > default local
    """
    # Try different environment variable names
    mongo_uri = (
        os.getenv('MONGO_URI') or 
        os.getenv('MONGODB_URL') or 
        os.getenv('MONGODB_URI') or
        'mongodb://localhost:27017/eaiser'  # Local fallback (trimmed)
    )
    
    # Database name from environment or default
    db_name = os.getenv('MONGODB_NAME', 'eaiser')
    
    logger.info(f"🔧 MongoDB Configuration:")
    logger.info(f"   URI: {mongo_uri[:50]}{'...' if len(mongo_uri) > 50 else ''}")
    logger.info(f"   Database: {db_name}")
    
    # Determine environment type
    if 'mongodb+srv://' in mongo_uri:
        logger.info(f"   Environment: Production (Atlas)")
        # Add TLS configuration for Atlas
        if 'tls=true' not in mongo_uri and 'ssl=true' not in mongo_uri:
            separator = '&' if '?' in mongo_uri else '?'
            mongo_uri += f"{separator}tls=true"
            logger.info(f"🔒 SSL/TLS enabled for secure connection")
    elif 'localhost' in mongo_uri or '127.0.0.1' in mongo_uri:
        logger.info(f"   Environment: Local Development")
    else:
        logger.info(f"   Environment: Custom/Remote")
    
    return mongo_uri, db_name

# Database connection setup with TLS-fixed configuration
mongo_uri, db_name = get_mongodb_config()

# Global variables for database connections
client = None
db = None
fs = None

# Store configuration globally for other functions
MONGO_URI = mongo_uri
DB_NAME = db_name

logger.info(f"🔧 MongoDB URI configured: {mongo_uri.split('@')[0].split('://')[0] if '@' in mongo_uri else mongo_uri.split('://')[0]}://***")
logger.info(f"📊 Database name: {db_name}")

# Parse database name from URI if not explicitly set
if not DB_NAME or DB_NAME == 'eaiser':
    try:
        # Parse the MongoDB URI safely to derive database name
        # This avoids NameError by using imported helpers directly
        parsed_uri = urlparse(MONGO_URI)
        if parsed_uri.path and len(parsed_uri.path) > 1:
            # Extract database name from URI path
            path_part = parsed_uri.path.strip('/')
            DB_NAME = path_part
        else:
            # For Atlas URIs, check if database is in query params
            query_params = parse_qs(parsed_uri.query)
            if 'authSource' in query_params:
                DB_NAME = query_params['authSource'][0]
            else:
                # Extract from URI path for standard format
                DB_NAME = parsed_uri.path[1:]  # Remove leading '/'
        
        DB_NAME = DB_NAME or "eaiser"  # Use environment or default
    except Exception as e:
        logger.warning(f"⚠️ Could not parse database name from URI: {e}")
        DB_NAME = DB_NAME or "eaiser"

logger.info(f"📊 Database name: {DB_NAME}")

# Global database connection
client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db = None
fs = None

async def init_db():
    """Initialize database connection with optimized connection pooling and error handling"""
    global client, db, fs
    
    # Check if we have a valid MongoDB URI
    if MONGO_URI == "mongodb://localhost:27017":
        logger.warning("⚠️ Using localhost MongoDB - this will fail in production!")
        logger.warning("⚠️ Please set MONGODB_URL environment variable for production deployment")
    
    try:
        logger.info(f"🔄 Attempting to connect to MongoDB...")
        logger.info(f"🔧 URI Type: {'Atlas Cloud' if 'mongodb+srv' in MONGO_URI else 'Local/Self-hosted'}")
        env = os.getenv("ENV", "development").lower()
        
        # Production-ready MongoDB client configuration
        client_config = {
            # Connection timeout settings - SIGNIFICANTLY INCREASED for slow networks
            # Handshake can take >3s on bad connections, so we give it 30s+
            "serverSelectionTimeoutMS": 30000,   # Wait 30s before giving up on finding a server
            "connectTimeoutMS": 45000,           # Wait 45s for initial connection (SSL handshake)
            "socketTimeoutMS": 60000,            # Wait 60s for operations to complete

            # Connection pooling for high performance
            "maxPoolSize": int(os.getenv("MONGO_POOL_MAXSIZE", "50")),
            "minPoolSize": int(os.getenv("MONGO_POOL_MINSIZE", "10")), # Increased default min pool to 10
            "maxIdleTimeMS": 120000,
            "waitQueueTimeoutMS": 30000,         # Wait 30s in queue before error

            # Performance optimizations
            "retryWrites": True,
            "w": "majority",
            "readPreference": "primaryPreferred",

            # Additional performance settings
            "compressors": "snappy,zlib",
            "zlibCompressionLevel": 6,
        }
        
        # Add SSL/TLS settings for Atlas connections
        if "mongodb+srv" in MONGO_URI or "ssl=true" in MONGO_URI:
            client_config.update({
                "tls": True,
                "tlsAllowInvalidCertificates": False,
            })
            logger.info("🔒 SSL/TLS enabled for secure connection")
        
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI, **client_config)
        
        # Test connection with retry logic
        max_retries = 3 if env != "production" else 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Connection attempt {attempt + 1}/{max_retries}...")
                
                # Use a shorter timeout for ping to fail fast
                await asyncio.wait_for(
                    client.admin.command('ping'),
                    timeout=5.0 if env != "production" else 10.0
                )
                
                logger.info(f"✅ MongoDB ping successful on attempt {attempt + 1}")
                break
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Connection attempt {attempt + 1} timed out")
                if attempt == max_retries - 1:
                    raise Exception("MongoDB connection timed out after all retries")
                    
            except Exception as e:
                logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                    
            # Exponential backoff with jitter
            delay = retry_delay * (2 ** attempt) + (attempt * 0.5)
            logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
            await asyncio.sleep(delay)
        
        # Initialize database and GridFS
        db = client[DB_NAME]
        fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db)
        
        # Verify database access
        collections = await db.list_collection_names()
        logger.info(f"✅ Successfully connected to MongoDB database: {DB_NAME}")
        logger.info(f"📊 Found {len(collections)} collections in database")
        logger.info(f"🔧 Connection pool: maxPoolSize=20, minPoolSize=2")
        
        # Create indexes for better performance
        try:
            # OPTIMIZATION: Run index creation in background to speed up startup
            # This prevents cold starts from hanging on index checks
            asyncio.create_task(create_indexes())
            logger.info("🚀 Database index creation scheduled in background")
        except Exception as e:
            logger.warning(f"⚠️ Index creation scheduling failed: {str(e)}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to connect to MongoDB: {error_msg}")
        
        # Provide specific guidance based on error type
        if "localhost:27017" in error_msg and "Connection refused" in error_msg:
            logger.error("💡 SOLUTION: Set MONGODB_URL environment variable to your MongoDB Atlas connection string")
            logger.error("💡 Example: MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/dbname")
            
        elif "authentication failed" in error_msg.lower():
            logger.error("💡 SOLUTION: Check your MongoDB username and password in the connection string")
            
        elif "timeout" in error_msg.lower():
            logger.error("💡 SOLUTION: Check network connectivity and MongoDB Atlas IP whitelist")
            
        # Don't raise exception to allow app to start without DB
        # This allows for graceful degradation
        logger.warning("⚠️ Application starting without MongoDB connection")
        logger.warning("⚠️ Database operations will fail until connection is restored")
        logger.warning("⚠️ Please fix MongoDB configuration and restart the application")
        
    return False

async def create_indexes() -> None:
    """Create minimal indexes required for core queries.

    This focuses on indexes actually referenced in code (e.g., hint
    "timestamp_desc") and common query paths to prevent startup warnings
    and improve performance.
    """
    global db
    if db is None:
        raise RuntimeError("Database is not initialized for index creation")

    try:
        issues = db["issues"]
        users = db["users"]

        # Index used by aggregation hint in get_issues()
        await issues.create_index([("timestamp", -1)], name="timestamp_desc")

        # Common filters and sorts
        await issues.create_index([("status", 1), ("timestamp", -1)], name="status_timestamp")
        await issues.create_index([("zip_code", 1)], name="zip_code")
        await issues.create_index([("issue_type", 1)], name="issue_type")
        await issues.create_index([("user_email", 1)], name="user_email")
        await issues.create_index([("latitude", 1), ("longitude", 1)], name="lat_lon")

        # Users collection unique email
        try:
            await users.create_index([("email", 1)], name="email_unique", unique=True)
        except Exception as e:
            # Code 85: IndexOptionsConflict - usually means it exists with different options or same options
            if "IndexOptionsConflict" in str(e) or "already exists" in str(e) or getattr(e, "code", 0) == 85:
                 logger.info(f"ℹ️ Index 'email_unique' already exists. Skipping.")
            else:
                 logger.warning(f"⚠️ Failed to create 'email_unique' index: {str(e)}")

        # Authority Mapping Review indexes
        authority_mapping_review = db["authority_mapping_review"]
        await authority_mapping_review.create_index([("resolved", 1)], name="resolved")
        await authority_mapping_review.create_index([("issue_type", 1)], name="issue_type")
        await authority_mapping_review.create_index([("flagged_at", -1)], name="flagged_at_desc")
        await authority_mapping_review.create_index([("case_id", 1)], name="case_id")

        logger.info("✅ Core MongoDB indexes created/verified")
    except Exception as e:
        logger.warning(f"⚠️ Index creation encountered an issue: {str(e)}")

async def close_db():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("🔒 MongoDB connection closed")

async def get_db():
    """Get the async database connection."""
    global db
    if db is None:
        await init_db()
    if db is None:
        raise RuntimeError("Async database connection could not be established")
    return db

async def get_fs():
    """Get the async GridFS instance."""
    global fs
    if fs is None:
        await init_db()
    if fs is None:
        raise RuntimeError("Async GridFS could not be initialized")
    return fs

async def store_issue(
    db,
    fs,
    issue_id: str,
    image_content: bytes,
    report: Dict[str, Any],
    unified_report: Optional[Dict[str, Any]],
    address: str,
    zip_code: Optional[str],
    latitude: float,
    longitude: float,
    issue_type: str,
    severity: str,
    category: str,
    priority: str,
    user_email: Optional[str],
    responsible_authorities: List[Dict[str, Any]],
    available_authorities: List[Dict[str, Any]]
) -> str:
    """
    Store an issue in MongoDB with zip code and return the image ID.
    """
    try:
        # Validate required fields
        required_fields = {
            "issue_type": issue_type,
            "severity": severity,
            "category": category,
            "priority": priority,
            "report": report
        }
        missing_fields = [k for k, v in required_fields.items() if not v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate zip code format (5-digit US zip code)
        if zip_code and not re.match(r"^\d{5}$", zip_code):
            logger.warning(f"Invalid zip code format for issue {issue_id}: {zip_code}. Setting to 'N/A'.")
            zip_code = "N/A"
        
        # Validate authority fields
        authority_email = [auth.get("email", "eaiser@momntumai.com") for auth in responsible_authorities]
        authority_name = [auth.get("name", "City Department") for auth in responsible_authorities]
        if not authority_email or not authority_name or None in authority_email or None in authority_name:
            authority_email = ["eaiser@momntumai.com"]
            authority_name = ["City Department"]
            logger.warning(f"No valid authorities provided for issue {issue_id}. Using defaults.")
        elif len(authority_email) != len(authority_name):
            raise ValueError("authority_email and authority_name lists must have the same length")
        
        # Validate available_authorities
        if available_authorities is not None:
            for auth in available_authorities:
                if not isinstance(auth, dict) or not all(key in auth for key in ["name", "email", "type"]):
                    logger.warning(f"Invalid available_authorities format for issue {issue_id}: {auth}. Setting to default.")
                    available_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
                    break
                if not auth.get("email") or not auth.get("name") or not auth.get("type"):
                    logger.warning(f"Missing required fields in available_authorities for issue {issue_id}: {auth}. Setting to default.")
                    available_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
                    break
        else:
            available_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
            logger.debug(f"No available_authorities provided for issue {issue_id}. Using default.")
        
        # Store the image in GridFS - FIXED: Properly await the upload operation
        try:
            image_id = await fs.upload_from_stream(
                filename=f"{issue_id}.jpg",
                source=image_content,
                metadata={"issue_id": issue_id}
            )
            logger.debug(f"Image uploaded successfully with ID: {image_id}")
        except Exception as e:
            logger.error(f"Failed to upload image for issue {issue_id}: {str(e)}", exc_info=True)
            raise
        
        # Create issue document with fallback values
        issue_document = {
            "_id": issue_id,
            "description": report.get("issue_overview", {}).get("summary_explanation", "No description provided"),
            "address": address or "Unknown Address",
            "zip_code": zip_code or "N/A",
            "latitude": latitude or 0.0,
            "longitude": longitude or 0.0,
            "issue_type": issue_type,
            "severity": severity,
            "image_id": str(image_id),  # Convert ObjectId to string
            "status": "pending",
            "report": report or {"message": "No report generated"},
            "unified_report": unified_report or {},
            "category": category,
            "priority": priority,
            "report_id": report.get("template_fields", {}).get("oid", ""),
            "timestamp": datetime.now().isoformat(),
            "authority_email": authority_email,
            "authority_name": authority_name,
            "timestamp_formatted": report.get("template_fields", {}).get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
            "timezone_name": report.get("template_fields", {}).get("timezone_name", "UTC"),
            "user_email": user_email,
            "available_authorities": available_authorities,
            "decline_reason": None,
            "decline_history": [],
            "email_status": "pending",
            "email_errors": []
        }
        
        # Insert issue into MongoDB
        try:
            await db.issues.insert_one(issue_document)
            
            # Invalidate issues cache since new issue was added
            redis_service = await get_redis_service()
            invalidated_count = await redis_service.invalidate_issues_cache()
            if invalidated_count > 0:
                logger.info(f"🧹 Invalidated {invalidated_count} cached issue entries after new issue creation")
            
            logger.info(
                f"Stored issue {issue_id} with image ID {image_id}, "
                f"authorities: {authority_name}, user_email: {user_email}, "
                f"zip_code: {zip_code or 'N/A'}, available_authorities: {available_authorities}"
            )
        except Exception as e:
            logger.error(f"Failed to insert issue document for {issue_id}: {str(e)}", exc_info=True)
            # Try to clean up the uploaded image if document insertion fails
            try:
                await fs.delete(image_id)
                logger.info(f"Cleaned up orphaned image {image_id} for issue {issue_id}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up orphaned image {image_id}: {str(cleanup_error)}", exc_info=True)
            raise
        
        return str(image_id)
    except Exception as e:
        logger.error(f"Failed to store issue {issue_id}: {str(e)}", exc_info=True)
        raise

async def update_pending_issue(issue_id: str, report: Dict[str, Any], decline_reason: str) -> bool:
    """
    Update a pending issue with a new report, decline reason, and append to decline history.
    """
    try:
        db = await get_db()
        # Validate inputs
        if not report:
            raise ValueError("Report cannot be empty")
        if not decline_reason:
            raise ValueError("Decline reason cannot be empty")
        
        # Append to decline_history
        decline_entry = {
            "reason": decline_reason,
            "timestamp": datetime.now().isoformat()
        }
        
        result = await db.issues.update_one(
            {"_id": issue_id, "status": "pending"},
            {
                "$set": {
                    "report": report,
                    "decline_reason": decline_reason,
                    "timestamp": datetime.now().isoformat()
                },
                "$push": {
                    "decline_history": decline_entry
                }
            }
        )
        
        if result.modified_count == 0:
            logger.warning(f"No pending issue found with ID {issue_id}")
            return False
        
        logger.info(f"Updated pending issue {issue_id} with decline reason: {decline_reason}")
        return True
    except Exception as e:
        logger.error(f"Failed to update pending issue {issue_id}: {str(e)}", exc_info=True)
        raise

async def get_issues(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve issues from MongoDB with Redis caching and pagination for optimized performance.
    
    Args:
        limit: Maximum number of issues to return (default: 50)
        skip: Number of issues to skip for pagination (default: 0)
    """
    try:
        # Try to get from Redis cache first
        redis_service = await get_redis_service()
        cached_issues = await redis_service.get_cached_issues_list(limit, skip)
        
        if cached_issues is not None:
            logger.info(f"🎯 Cache HIT: Retrieved {len(cached_issues)} issues from Redis (limit: {limit}, skip: {skip})")
            return cached_issues
        
        # Cache miss - fetch from MongoDB
        logger.info(f"❌ Cache MISS: Fetching from MongoDB (limit: {limit}, skip: {skip})")
        db = await get_db()

        # Read a defensible per-list timeout from environment (ms)
        # Comment: Keeps list queries from hanging indefinitely under load
        LIST_TIMEOUT_MS = int(os.getenv("LIST_TIMEOUT_MS", "5000"))
        LIST_TIMEOUT_S = max(1.0, LIST_TIMEOUT_MS / 1000.0)
        
        # Use projection to only fetch required fields for better performance
        projection = {
            "_id": 1,
            "issue_type": 1,
            "description": 1,
            "address": 1,
            "zip_code": 1,
            "latitude": 1,
            "longitude": 1,
            "severity": 1,
            "category": 1,
            "priority": 1,
            "user_email": 1,
            "status": 1,
            "timestamp": 1,
            "timestamp_formatted": 1,
            "timezone_name": 1,
            "authority_email": 1,
            "authority_name": 1,
            "image_id": 1,
            "decline_reason": 1,
            "decline_history": 1,
            "available_authorities": 1
        }
        
        # Optimized aggregation pipeline with index hints for maximum performance
        pipeline = [
            # Use index hint to force usage of timestamp_desc index
            {"$match": {}},  # Empty match to allow index hinting
            {"$sort": {"timestamp": -1}},  # Sort by timestamp descending (uses timestamp_desc index)
            {"$skip": skip},
            {"$limit": limit},
            {"$project": projection}
        ]

        async def _run_aggregate():
            """Run aggregate with defensive options and return list.
            Comment: Encapsulates per-query limits and prevents hanging operations.
            """
            # Add index hint and per-query timeout for better performance and resilience
            cursor = db.issues.aggregate(
                pipeline,
                hint="timestamp_desc",
                maxTimeMS=LIST_TIMEOUT_MS,  # Abort server-side if query runs too long
                allowDiskUse=False,         # Prevent disk use for this lightweight listing
                batchSize=500               # Stream results in reasonable batches
            )
            return await cursor.to_list(length=limit)  # Respect explicit length

        # Execute with cooperative client-side timeout
        start_ts = time.time()
        try:
            issues = await asyncio.wait_for(_run_aggregate(), timeout=LIST_TIMEOUT_S)
        except asyncio.TimeoutError:
            elapsed = (time.time() - start_ts) * 1000
            logger.warning(f"⚠️ MongoDB list timed out after {elapsed:.2f} ms (limit={limit}, skip={skip})")
            issues = []
        except Exception as e:
            logger.error(f"❌ MongoDB list failed (limit={limit}, skip={skip}): {str(e)}", exc_info=True)
            issues = []
        
        for issue in issues:
            # Minimal processing - only set defaults for missing fields
            issue.setdefault("issue_type", "Unknown Issue")
            issue.setdefault("description", "No description")
            issue.setdefault("address", "Unknown Address")
            issue.setdefault("zip_code", "N/A")
            issue.setdefault("latitude", 0.0)
            issue.setdefault("longitude", 0.0)
            issue.setdefault("severity", "Medium")
            issue.setdefault("category", "Public")
            issue.setdefault("priority", "Medium")
            issue.setdefault("decline_reason", None)
            issue.setdefault("decline_history", [])
            issue.setdefault("available_authorities", [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}])
            issue.setdefault("timestamp_formatted", datetime.now().strftime("%Y-%m-%d %H:%M"))
            issue.setdefault("timezone_name", "UTC")
            
            # Optimized authority field processing
            authority_email = issue.get("authority_email", ["eaiser@momntumai.com"])
            if not isinstance(authority_email, list):
                authority_email = [str(authority_email)] if authority_email else ["eaiser@momntumai.com"]
            issue["authority_email"] = authority_email
            
            authority_name = issue.get("authority_name", ["City Department"])
            if not isinstance(authority_name, list):
                 authority_name = [str(authority_name)]
            issue["authority_name"] = authority_name

            # Format MongoDB ID as string
            issue["_id"] = str(issue["_id"])

        return issues
        
    except Exception as e:
        logger.error(f"Failed to retrieve issues: {str(e)}")
        return []

async def get_user_issues(user_email: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve issues for a specific user from MongoDB.
    """
    try:
        db = await get_db()
        LIST_TIMEOUT_MS = int(os.getenv("LIST_TIMEOUT_MS", "5000"))
        LIST_TIMEOUT_S = max(1.0, LIST_TIMEOUT_MS / 1000.0)
        
        projection = {
            "_id": 1,
            "issue_type": 1,
            "description": 1,
            "address": 1,
            "zip_code": 1,
            "latitude": 1,
            "longitude": 1,
            "severity": 1,
            "category": 1,
            "priority": 1,
            "user_email": 1,
            "status": 1,
            "timestamp": 1,
            "timestamp_formatted": 1,
            "timezone_name": 1,
            "authority_email": 1,
            "authority_name": 1,
            "image_id": 1,
            "decline_reason": 1,
            "decline_history": 1,
            "available_authorities": 1
        }
        
        # Filter by user_email
        pipeline = [
            {"$match": {"user_email": user_email}},
            {"$sort": {"timestamp": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {"$project": projection}
        ]

        async def _run_aggregate_user():
            cursor = db.issues.aggregate(
                pipeline,
                hint="timestamp_desc",
                maxTimeMS=LIST_TIMEOUT_MS,
                allowDiskUse=False,
                batchSize=500
            )
            return await cursor.to_list(length=limit)

        try:
            issues = await asyncio.wait_for(_run_aggregate_user(), timeout=LIST_TIMEOUT_S)
        except asyncio.TimeoutError:
            issues = []
        except Exception as e:
            logger.error(f"❌ MongoDB user list failed: {str(e)}", exc_info=True)
            issues = []
        
        # Process defaults (same as get_issues)
        for issue in issues:
            issue["_id"] = str(issue["_id"])
            issue.setdefault("issue_type", "Unknown Issue")
            issue.setdefault("description", "No description")
            issue.setdefault("address", "Unknown Address")
            issue.setdefault("zip_code", "N/A")
            issue.setdefault("severity", "Medium")
            issue.setdefault("category", "Public")
            issue.setdefault("priority", "Medium")
            issue.setdefault("status", "pending")
            issue.setdefault("timestamp_formatted", datetime.now().strftime("%Y-%m-%d %H:%M"))

        return issues

    except Exception as e:
        logger.error(f"Failed to get user issues: {e}")
        return []

# Legacy function for backward compatibility
async def get_all_issues() -> List[Dict[str, Any]]:
    """
    Retrieve all issues from MongoDB (legacy function - use get_issues with pagination instead).
    """
    return await get_issues(limit=1000, skip=0)  # Reasonable limit for backward compatibility

async def get_report(issue_id: str) -> Dict[str, Any]:
    """
    Retrieve a single issue by ID.
    """
    try:
        db = await get_db()
        issue = await db.issues.find_one({"_id": issue_id})
        if not issue:
            logger.warning(f"No issue found with ID {issue_id}")
            return None
        
        # Ensure default values and list conversion
        issue["issue_type"] = issue.get("issue_type", "Unknown Issue")
        issue["description"] = issue.get("description", "No description")
        issue["address"] = issue.get("address", "Unknown Address")
        issue["zip_code"] = issue.get("zip_code", "N/A")
        issue["latitude"] = issue.get("latitude", 0.0)
        issue["longitude"] = issue.get("longitude", 0.0)
        issue["severity"] = issue.get("severity", "Medium")
        issue["category"] = issue.get("category", "Public")
        issue["priority"] = issue.get("priority", "Medium")
        issue["user_email"] = issue.get("user_email", None)
        issue["decline_reason"] = issue.get("decline_reason", None)
        issue["decline_history"] = issue.get("decline_history", [])
        issue["available_authorities"] = issue.get("available_authorities", [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}])
        
        # Clean authority_email
        authority_email = issue.get("authority_email", ["eaiser@momntumai.com"])
        if isinstance(authority_email, list):
            authority_email = [str(email) for email in authority_email if email is not None and isinstance(email, str)]
            if not authority_email:
                authority_email = ["eaiser@momntumai.com"]
        else:
            authority_email = [str(authority_email)] if authority_email else ["eaiser@momntumai.com"]
        issue["authority_email"] = authority_email
        
        # Clean authority_name
        authority_name = issue.get("authority_name", ["City Department"])
        if isinstance(authority_name, list):
            authority_name = [str(name) for name in authority_name if name is not None and isinstance(name, str)]
            if not authority_name:
                authority_name = ["City Department"]
        else:
            authority_name = [str(authority_name)] if authority_name else ["City Department"]
        issue["authority_name"] = authority_name
        
        issue["timestamp_formatted"] = issue.get("timestamp_formatted", datetime.now().strftime("%Y-%m-%d %H:%M"))
        issue["timezone_name"] = issue.get("timezone_name", "UTC")
        
        # Validate image_id
        image_id = issue.get("image_id")
        if image_id and not isinstance(image_id, str):
            logger.warning(f"Invalid image_id format for issue {issue_id}: {type(image_id)}. Converting to string.")
            issue["image_id"] = str(image_id)
        
        logger.info(f"Retrieved issue {issue_id}")
        return issue
    except Exception as e:
        logger.error(f"Failed to retrieve issue {issue_id}: {str(e)}", exc_info=True)
        raise

async def update_issue_status(issue_id: str, status: str) -> bool:
    """
    Update the status of an issue.
    """
    try:
        db = await get_db()
        valid_statuses = ["pending", "accepted", "rejected", "completed", "needs_review"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of {valid_statuses}")
        
        result = await db.issues.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    # Clear decline_reason and decline_history on accept
                    "decline_reason": None if status == "accepted" else None,
                    "decline_history": [] if status == "accepted" else []
                }
            }
        )
        
        if result.modified_count == 0:
            logger.warning(f"No issue found with ID {issue_id}")
            return False
        
        logger.info(f"Updated status for issue {issue_id} to {status}")
        return True
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id} status: {str(e)}", exc_info=True)
        raise

async def update_issue(issue_id: str, update_data: dict) -> bool:
    """
    Update an existing issue in MongoDB and invalidate cache.
    
    Args:
        issue_id (str): The ID of the issue to update
        update_data (dict): Data to update
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        db = await get_db()
        
        # Add updated timestamp
        update_data['updated_at'] = datetime.now()
        
        result = await db.issues.update_one(
            {"_id": issue_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            # Invalidate issues cache since issue was updated
            redis_service = await get_redis_service()
            invalidated_count = await redis_service.invalidate_issues_cache()
            if invalidated_count > 0:
                logger.info(f"🧹 Invalidated {invalidated_count} cached issue entries after issue update")
            
            logger.info(f"Issue {issue_id} updated successfully")
            return True
        else:
            logger.warning(f"No issue found with ID {issue_id} or no changes made")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update issue {issue_id}: {str(e)}", exc_info=True)
        return False

async def get_image(issue_id: str) -> bytes:
    """
    Retrieve image content from GridFS by issue ID.
    """
    try:
        fs = await get_fs()
        # Find the image in GridFS
        gridout = await fs.open_download_stream_by_name(f"{issue_id}.jpg")
        image_content = await gridout.read()
        return image_content
    except gridfs.errors.NoFile:
        logger.error(f"Image not found for issue {issue_id}")
        raise HTTPException(status_code=404, detail=f"Image not found for issue {issue_id}")
    except Exception as e:
        logger.error(f"Failed to retrieve image for issue {issue_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")