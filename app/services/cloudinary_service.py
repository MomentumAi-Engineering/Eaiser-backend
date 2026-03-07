import os
import logging
import cloudinary
import cloudinary.uploader
import cloudinary.api
from fastapi import UploadFile
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try initializing cloudinary if configs are available
try:
    # Ensure they are loaded
    from dotenv import load_dotenv
    from pathlib import Path
    try:
        current_dir = Path(__file__).parent.parent.absolute()
        load_dotenv(current_dir / ".env")
    except:
        pass

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")

    if cloud_name and api_key and api_secret:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
        logger.info("✅ Cloudinary initialized successfully.")
    else:
        logger.warning("⚠️ Cloudinary variables not set in .env. Uploads will fail if called.")
except Exception as e:
    logger.error(f"Failed to initialize Cloudinary: {e}")

async def upload_file_to_cloudinary(file: Optional[UploadFile] = None, contents: Optional[bytes] = None, folder: str = "chat_media") -> Optional[Dict]:
    """
    Uploads a file or bytes to Cloudinary and returns the URL and Type
    """
    if not os.getenv("CLOUDINARY_CLOUD_NAME"):
        logger.error("Cloudinary is not configured!")
        return None

    try:
        resource_type = "image"
        
        # Read the file data if file provided
        if file:
            file_type = file.content_type
            resource_type = "video" if file_type and file_type.startswith("video") else "image"
            contents = await file.read()
        
        if not contents:
            logger.error("No file or contents provided for Cloudinary upload")
            return None

        # Upload using the raw file content bytes
        upload_result = cloudinary.uploader.upload(
            contents,
            folder=f"eaiser/{folder}",
            resource_type=resource_type
        )

        if file:
            await file.seek(0)

        return {
            "url": upload_result.get("secure_url"),
            "public_id": upload_result.get("public_id"),
            "format": upload_result.get("format"),
            "type": resource_type
        }

    except Exception as e:
        logger.error(f"❌ Cloudinary Upload Error: {str(e)}")
        return None
