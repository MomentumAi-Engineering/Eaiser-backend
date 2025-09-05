import asyncio
import json
import re
import io
from datetime import datetime
from typing import Dict, Any, Optional
import pytz
from PIL import Image
import google.generativeai as genai
import logging
import json
import os
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# Simple data loader function
async def load_json_data(filename: str) -> dict:
    """Load JSON data from file"""
    try:
        file_path = Path(__file__).parent.parent / "data" / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"JSON file not found: {filename}")
            return {}
    except Exception as e:
        logger.error(f"Error loading JSON file {filename}: {e}")
        return {}
from app.utils.location import get_authority_by_zip_code, get_authority
from app.utils.timezone import get_timezone_name
import redis
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor

# Redis connection for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    logger.info("Redis connected for AI service caching")
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available for AI service: {e}")

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

# Cache TTL settings
CACHE_TTL = {
    'department_mapping': 3600,  # 1 hour
    'authority_data': 1800,      # 30 minutes
    'timezone_data': 7200,       # 2 hours
    'ai_report': 300             # 5 minutes for similar reports
}

async def get_cached_data(cache_key: str, ttl: int = 300) -> Optional[Any]:
    """Get data from Redis cache"""
    if not redis_client:
        return None
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return pickle.loads(cached_data)
    except Exception as e:
        logger.warning(f"Cache get error for {cache_key}: {e}")
    return None

async def set_cached_data(cache_key: str, data: Any, ttl: int = 300) -> None:
    """Set data in Redis cache"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(cache_key, ttl, pickle.dumps(data))
    except Exception as e:
        logger.warning(f"Cache set error for {cache_key}: {e}")

def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key from parameters"""
    key_data = f"{prefix}:" + ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return hashlib.md5(key_data.encode()).hexdigest()

async def get_department_mapping_cached() -> Dict[str, Any]:
    """Get department mapping with caching"""
    cache_key = "department_mapping"
    cached_data = await get_cached_data(cache_key, CACHE_TTL['department_mapping'])
    
    if cached_data:
        return cached_data
    
    # Load from file
    department_mapping = await load_json_data("issue_department_map.json")
    await set_cached_data(cache_key, department_mapping, CACHE_TTL['department_mapping'])
    return department_mapping

async def get_authority_data_cached(zip_code: Optional[str], address: str, issue_type: str, 
                                  latitude: float, longitude: float, category: str) -> Dict[str, Any]:
    """Get authority data with caching"""
    cache_key = generate_cache_key(
        "authority", 
        zip_code=zip_code or "", 
        address=address or "", 
        issue_type=issue_type, 
        category=category
    )
    
    cached_data = await get_cached_data(cache_key, CACHE_TTL['authority_data'])
    if cached_data:
        return cached_data
    
    # Get authority data
    if zip_code:
        authority_data = await asyncio.to_thread(get_authority_by_zip_code, zip_code, issue_type, category)
    else:
        authority_data = await asyncio.to_thread(get_authority, address, issue_type, latitude, longitude, category)
    
    await set_cached_data(cache_key, authority_data, CACHE_TTL['authority_data'])
    return authority_data

async def get_timezone_cached(latitude: float, longitude: float) -> str:
    """Get timezone with caching"""
    cache_key = generate_cache_key("timezone", lat=round(latitude, 4), lng=round(longitude, 4))
    
    cached_data = await get_cached_data(cache_key, CACHE_TTL['timezone_data'])
    if cached_data:
        return cached_data
    
    # Get timezone
    timezone_str = await asyncio.to_thread(get_timezone_name, latitude, longitude) or "UTC"
    await set_cached_data(cache_key, timezone_str, CACHE_TTL['timezone_data'])
    return timezone_str

async def generate_ai_report_async(prompt: str, image_content: bytes) -> str:
    """Generate AI report asynchronously"""
    def _generate_report():
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt, Image.open(io.BytesIO(image_content))])
        return response.text
    
    # Run in thread pool to avoid blocking
    return await asyncio.get_event_loop().run_in_executor(executor, _generate_report)

async def generate_report_optimized(
    image_content: bytes,
    description: str,
    issue_type: str,
    severity: str,
    address: str,
    zip_code: Optional[str],
    latitude: float,
    longitude: float,
    issue_id: str,
    confidence: float,
    category: str,
    priority: str,
    decline_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Optimized report generation with caching and async processing"""
    
    # Generate cache key for similar reports
    report_cache_key = generate_cache_key(
        "ai_report",
        issue_type=issue_type,
        severity=severity,
        category=category,
        description_hash=hashlib.md5(description.encode()).hexdigest()[:8]
    )
    
    # Check if similar report exists in cache
    cached_report = await get_cached_data(report_cache_key, CACHE_TTL['ai_report'])
    if cached_report and not decline_reason:  # Don't use cache for declined reports
        logger.info(f"Using cached report for similar issue: {issue_id}")
        # Update dynamic fields
        cached_report["template_fields"]["oid"] = f"SNAPFIX-{datetime.now().year}-{str(int(datetime.now().strftime('%Y%m%d%H%M%S')) % 1000000).zfill(6)}"
        cached_report["additional_notes"] = cached_report["additional_notes"].replace(
            cached_report["additional_notes"].split("Issue ID: ")[1].split(".")[0],
            issue_id
        )
        return cached_report
    
    # Prepare location string
    location_str = (
        f"{address}, {zip_code}" if address and address.lower() != "not specified" and zip_code
        else address if address and address.lower() != "not specified"
        else f"Coordinates: {latitude}, {longitude}" if latitude and longitude
        else "Unknown Location"
    )
    
    # Start all async operations concurrently
    tasks = {
        'department_mapping': get_department_mapping_cached(),
        'authority_data': get_authority_data_cached(zip_code, address, issue_type, latitude, longitude, category),
        'timezone': get_timezone_cached(latitude, longitude)
    }
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # Handle results
    department_mapping = results[0] if not isinstance(results[0], Exception) else {}
    authority_data = results[1] if not isinstance(results[1], Exception) else {}
    timezone_str = results[2] if not isinstance(results[2], Exception) else "UTC"
    
    # Resolve department
    normalized_issue_type = issue_type.lower().replace(" ", "_")
    department = "general"
    
    if normalized_issue_type in department_mapping:
        department = department_mapping[normalized_issue_type][0]
    elif issue_type.lower() in department_mapping:
        department = department_mapping[issue_type.lower()][0]
    elif issue_type in department_mapping:
        department = department_mapping[issue_type][0]
    
    logger.info(f"Resolved department for issue type '{issue_type}': {department}")
    
    # Process authority data
    responsible_authorities = authority_data.get("responsible_authorities", 
        [{"name": department, "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])
    available_authorities = authority_data.get("available_authorities", 
        [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])
    
    if available_authorities and "message" in available_authorities[0]:
        available_authorities = [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}]
    
    # Generate timestamps
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    local_time = now.strftime("%Y-%m-%d %H:%M")
    utc_time = now.astimezone(pytz.UTC).strftime("%H:%M")
    report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
    report_id = f"SNAPFIX-{now.year}-{report_number}"
    image_filename = f"IMG1_{now.strftime('%Y%m%d_%H%M')}.jpg"
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    
    # Try AI generation with retries
    for attempt in range(2):  # Reduced from 3 to 2 attempts for faster response
        try:
            # Prepare optimized prompt
            decline_prompt = f"- Decline Reason: {decline_reason}\n" if decline_reason else ""
            zip_code_prompt = f"- Zip Code: {zip_code}\n" if zip_code else ""
            
            prompt = f"""
You are an AI assistant for SnapFix AI. Generate a concise infrastructure issue report in JSON format.
Input:
- Issue Type: {issue_type.title()}
- Severity: {severity}
- Confidence: {confidence:.1f}%
- Description: {description}
- Category: {category}
- Location: {location_str}
- Issue ID: {issue_id}
- Department: {department}
- Map: {map_link}
- Priority: {priority}
{decline_prompt}{zip_code_prompt}

Return this JSON structure (no markdown, no explanation):
{{
  "issue_overview": {{
    "issue_type": "{issue_type.title()}",
    "severity": "{severity.lower()}",
    "confidence": {confidence},
    "category": "{category}",
    "summary_explanation": "Brief 2-3 line explanation based on image and description. Include location and zip code if provided."
  }},
  "detailed_analysis": {{
    "root_causes": "Likely causes",
    "potential_consequences_if_ignored": "Risks if not addressed",
    "public_safety_risk": "low|medium|high",
    "environmental_impact": "low|medium|high|none",
    "structural_implications": "low|medium|high|none",
    "legal_or_regulatory_considerations": "Relevant regulations or null",
    "feedback": {f'"User decline reason: {decline_reason}"' if decline_reason else 'null'}
  }},
  "recommended_actions": ["Action 1 with timeframe", "Action 2 with timeframe"],
  "template_fields": {{
    "oid": "{report_id}",
    "timestamp": "{local_time}",
    "utc_time": "{utc_time}",
    "priority": "{priority}",
    "tracking_link": "https://momentum-ai.org/track/{report_id}",
    "image_filename": "{image_filename}",
    "ai_tag": "{issue_type.title()}",
    "app_version": "1.5.3",
    "device_type": "Mobile (Generic)",
    "map_link": "{map_link}",
    "zip_code": "{zip_code if zip_code else 'N/A'}"
  }}
}}
Keep response under 150 words, professional tone.
"""
            
            # Generate AI report asynchronously
            response_text = await generate_ai_report_async(prompt, image_content)
            logger.info(f"AI report generated (attempt {attempt + 1}) for issue {issue_id}")
            
            # Extract and validate JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            
            json_text = json_match.group(0)
            report = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["issue_overview", "detailed_analysis", "recommended_actions", "template_fields"]
            missing_fields = [field for field in required_fields if field not in report]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Add additional fields
            report["responsible_authorities_or_parties"] = responsible_authorities
            report["available_authorities"] = available_authorities
            report["additional_notes"] = f"Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}."
            
            # Cache successful report (only if not declined)
            if not decline_reason:
                await set_cached_data(report_cache_key, report, CACHE_TTL['ai_report'])
            
            logger.info(f"Optimized report generated successfully for issue {issue_id}")
            return report
            
        except Exception as e:
            logger.warning(f"AI generation attempt {attempt + 1} failed: {str(e)}")
            if attempt == 1:  # Last attempt
                logger.error(f"AI generation failed after 2 attempts: {str(e)}")
                break
            await asyncio.sleep(0.5)  # Shorter wait time
    
    # Fast fallback report (pre-computed templates)
    logger.info(f"Using fallback report for issue {issue_id}")
    
    # Issue-specific quick actions
    action_templates = {
        "fire": ["Dispatch fire department immediately", "Evacuate area if necessary"],
        "pothole": ["Fill pothole within 48 hours", "Mark area with safety cones"],
        "garbage": ["Remove debris within 24 hours", "Install additional bins"],
        "flood": ["Deploy pumps immediately", "Clear drainage within 12 hours"],
        "vandalism": ["Remove graffiti within 72 hours", "Increase security patrols"],
        "structural_damage": ["Cordon off area immediately", "Conduct inspection within 24 hours"],
        "property_damage": ["Assess damage extent", "Contact property owner within 24 hours"]
    }
    
    actions = action_templates.get(issue_type.lower(), [f"Notify {department} for action", "Conduct professional inspection"])
    
    fallback_report = {
        "issue_overview": {
            "issue_type": issue_type.title(),
            "severity": severity.lower(),
            "confidence": confidence,
            "category": category,
            "summary_explanation": f"AI identified a {issue_type} at {location_str} based on: {description}."
        },
        "detailed_analysis": {
            "root_causes": "Requires inspection to determine exact cause",
            "potential_consequences_if_ignored": "May pose safety or compliance risks",
            "public_safety_risk": severity.lower(),
            "environmental_impact": "low",
            "structural_implications": "medium" if issue_type in ["structural_damage", "property_damage"] else "low",
            "legal_or_regulatory_considerations": "Local regulations may apply",
            "feedback": f"User-provided decline reason: {decline_reason}" if decline_reason else None
        },
        "recommended_actions": actions,
        "responsible_authorities_or_parties": responsible_authorities,
        "available_authorities": available_authorities,
        "additional_notes": f"Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}.",
        "template_fields": {
            "oid": report_id,
            "timestamp": local_time,
            "utc_time": utc_time,
            "priority": priority,
            "tracking_link": f"https://momentum-ai.org/track/{report_id}",
            "image_filename": image_filename,
            "ai_tag": issue_type.title(),
            "app_version": "1.5.3",
            "device_type": "Mobile (Generic)",
            "map_link": map_link,
            "zip_code": zip_code if zip_code else "N/A"
        }
    }
    
    if decline_reason:
        fallback_report["issue_overview"]["summary_explanation"] += f" Declined due to: {decline_reason}."
    
    logger.info(f"Fallback report generated for issue {issue_id}")
    return fallback_report

# Background task for pre-warming cache
async def prewarm_cache():
    """Pre-warm frequently used cache data"""
    try:
        # Pre-load department mapping
        await get_department_mapping_cached()
        logger.info("Cache pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Cache pre-warming failed: {e}")

# Note: Cache will be prewarmed when first function is called
# Cannot use asyncio.create_task at module level