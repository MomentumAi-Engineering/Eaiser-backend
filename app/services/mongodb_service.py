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
from urllib.parse import urlparse

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

# Database connection setup
MONGO_URI = os.getenv("MONGODB_URL", "mongodb+srv://snapfix:Chrishabh100@cluster0.mk0k1wa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.getenv("MONGODB_NAME", "snapfix")

# Parse database name from MONGO_URI if provided
parsed_uri = urlparse(MONGO_URI)
if parsed_uri.path and parsed_uri.path.strip("/"):
    DB_NAME = parsed_uri.path.strip("/")

# Global database connection
client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db = None
fs = None

async def initialize_db(max_retries=3, retry_delay=5):
    """Initialize the MongoDB async connection and GridFS with retry logic."""
    global client, db, fs
    for attempt in range(max_retries):
        try:
            client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            await client.admin.command('ping')
            db = client[DB_NAME]
            fs = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db)
            logger.info(f"Async database connection established to {DB_NAME}")
            return
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error("Max retries reached. Could not connect to MongoDB.")
                raise RuntimeError(f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB initialization: {str(e)}")
            raise

async def get_db():
    """Get the async database connection."""
    global db
    if db is None:
        await initialize_db()
    if db is None:
        raise RuntimeError("Async database connection could not be established")
    return db

async def get_fs():
    """Get the async GridFS instance."""
    global fs
    if fs is None:
        await initialize_db()
    if fs is None:
        raise RuntimeError("Async GridFS could not be initialized")
    return fs

async def store_issue(
    db,
    fs,
    issue_id: str,
    image_content: bytes,
    report: Dict[str, Any],
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
        authority_email = [auth.get("email", "snapfix@momntumai.com") for auth in responsible_authorities]
        authority_name = [auth.get("name", "City Department") for auth in responsible_authorities]
        if not authority_email or not authority_name or None in authority_email or None in authority_name:
            authority_email = ["snapfix@momntumai.com"]
            authority_name = ["City Department"]
            logger.warning(f"No valid authorities provided for issue {issue_id}. Using defaults.")
        elif len(authority_email) != len(authority_name):
            raise ValueError("authority_email and authority_name lists must have the same length")
        
        # Validate available_authorities
        if available_authorities is not None:
            for auth in available_authorities:
                if not isinstance(auth, dict) or not all(key in auth for key in ["name", "email", "type"]):
                    logger.warning(f"Invalid available_authorities format for issue {issue_id}: {auth}. Setting to default.")
                    available_authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
                    break
                if not auth.get("email") or not auth.get("name") or not auth.get("type"):
                    logger.warning(f"Missing required fields in available_authorities for issue {issue_id}: {auth}. Setting to default.")
                    available_authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
                    break
        else:
            available_authorities = [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}]
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

async def get_issues() -> List[Dict[str, Any]]:
    """
    Retrieve all issues from MongoDB.
    """
    try:
        db = await get_db()
        issues = []
        
        async for issue in db.issues.find().sort("timestamp", -1):
            # Ensure default values and list conversion for authority fields
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
            issue["available_authorities"] = issue.get("available_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])
            
            # Clean authority_email
            authority_email = issue.get("authority_email", ["snapfix@momntumai.com"])
            if isinstance(authority_email, list):
                authority_email = [str(email) for email in authority_email if email is not None and isinstance(email, str)]
                if not authority_email:
                    authority_email = ["snapfix@momntumai.com"]
            else:
                authority_email = [str(authority_email)] if authority_email else ["snapfix@momntumai.com"]
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
            
            issues.append(issue)
        
        logger.info(f"Retrieved {len(issues)} issues from MongoDB")
        return issues
    except Exception as e:
        logger.error(f"Failed to retrieve issues: {str(e)}", exc_info=True)
        raise

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
        issue["available_authorities"] = issue.get("available_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])
        
        # Clean authority_email
        authority_email = issue.get("authority_email", ["snapfix@momntumai.com"])
        if isinstance(authority_email, list):
            authority_email = [str(email) for email in authority_email if email is not None and isinstance(email, str)]
            if not authority_email:
                authority_email = ["snapfix@momntumai.com"]
        else:
            authority_email = [str(authority_email)] if authority_email else ["snapfix@momntumai.com"]
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
        valid_statuses = ["pending", "accepted", "rejected", "completed"]
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