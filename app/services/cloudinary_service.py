import os
import logging
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
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


# ---------------------------------------------------------------------------
# 🚀 CDN-Optimized Image Transformations
# ---------------------------------------------------------------------------

# Pre-defined transformation profiles for different use cases
CDN_TRANSFORMS = {
    "thumbnail": {
        "width": 200, "height": 200, "crop": "fill",
        "quality": "auto:low", "fetch_format": "auto",
    },
    "card": {
        "width": 400, "height": 300, "crop": "fill",
        "quality": "auto:good", "fetch_format": "auto",
    },
    "detail": {
        "width": 800, "height": 600, "crop": "limit",
        "quality": "auto:good", "fetch_format": "auto",
    },
    "full": {
        "width": 1200, "crop": "limit",
        "quality": "auto:best", "fetch_format": "auto",
    },
    "report": {
        "width": 1024, "crop": "limit",
        "quality": "auto:good", "fetch_format": "auto",
    },
}


def get_optimized_url(url: str, profile: str = "detail") -> str:
    """
    Convert a raw Cloudinary URL into a CDN-optimized URL.
    
    Profiles: thumbnail, card, detail, full, report
    
    Example:
      Input:  https://res.cloudinary.com/dpw92yv54/image/upload/v123/eaiser/report_images/abc.jpg
      Output: https://res.cloudinary.com/dpw92yv54/image/upload/w_800,h_600,c_limit,q_auto:good,f_auto/v123/eaiser/report_images/abc.jpg
    """
    if not url or "cloudinary" not in url:
        return url
    
    transform = CDN_TRANSFORMS.get(profile, CDN_TRANSFORMS["detail"])
    
    try:
        # Extract public_id from URL
        # URL format: .../image/upload/v{version}/{public_id}.{ext}
        parts = url.split("/upload/")
        if len(parts) != 2:
            return url
        
        base = parts[0] + "/upload/"
        path = parts[1]
        
        # Build transformation string
        t_parts = []
        if "width" in transform:
            t_parts.append(f"w_{transform['width']}")
        if "height" in transform:
            t_parts.append(f"h_{transform['height']}")
        if "crop" in transform:
            t_parts.append(f"c_{transform['crop']}")
        if "quality" in transform:
            t_parts.append(f"q_{transform['quality']}")
        if "fetch_format" in transform:
            t_parts.append(f"f_{transform['fetch_format']}")
        
        transform_str = ",".join(t_parts)
        return f"{base}{transform_str}/{path}"
    except Exception:
        return url


def get_responsive_urls(url: str) -> Dict[str, str]:
    """
    Generate multiple size variants for responsive image loading.
    Frontend can use srcset to load the right size.
    """
    return {
        "thumbnail": get_optimized_url(url, "thumbnail"),
        "card": get_optimized_url(url, "card"),
        "detail": get_optimized_url(url, "detail"),
        "full": get_optimized_url(url, "full"),
    }


async def upload_file_to_cloudinary(file: Optional[UploadFile] = None, contents: Optional[bytes] = None, folder: str = "chat_media") -> Optional[Dict]:
    """
    Uploads a file or bytes to Cloudinary and returns the URL and Type.
    🚀 Now includes CDN-optimized URLs for faster frontend loading.
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

        import asyncio
        # Upload using the raw file content bytes in a separate thread because it is blocking I/O!
        # 🚀 Added eager transformations for pre-generating CDN variants
        upload_options = {
            "folder": f"eaiser/{folder}",
            "resource_type": resource_type,
        }
        
        # For images, add CDN optimizations
        if resource_type == "image":
            upload_options["eager"] = [
                {"width": 400, "height": 300, "crop": "fill", "quality": "auto:good", "fetch_format": "auto"},
                {"width": 200, "height": 200, "crop": "fill", "quality": "auto:low", "fetch_format": "auto"},
            ]
            upload_options["eager_async"] = True  # Generate in background
        
        upload_result = await asyncio.to_thread(
            cloudinary.uploader.upload,
            contents,
            **upload_options
        )

        if file:
            await file.seek(0)

        raw_url = upload_result.get("secure_url")
        
        result = {
            "url": raw_url,
            "public_id": upload_result.get("public_id"),
            "format": upload_result.get("format"),
            "type": resource_type,
        }
        
        # 🚀 Add CDN-optimized URLs for images
        if resource_type == "image" and raw_url:
            result["cdn_urls"] = get_responsive_urls(raw_url)
            result["optimized_url"] = get_optimized_url(raw_url, "detail")
        
        return result

    except Exception as e:
        logger.error(f"❌ Cloudinary Upload Error: {str(e)}")
        return None

