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

# Load environment variables and configure Gemini
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set; disabling AI features.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to configure Gemini API: {e}. Disabling AI features.")
        GEMINI_API_KEY = None

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
from utils.location import get_authority_by_zip_code, get_authority
from utils.timezone import get_timezone_name
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

# Add a safe model getter with fallbacks

def get_gemini_model():
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        logger.warning(f"{model_name} not available for current API version; attempting fallbacks: {e}")
        for alt in [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.0-pro-vision",
            "gemini-1.0-pro"
        ]:
            try:
                logger.info(f"Trying fallback model: {alt}")
                return genai.GenerativeModel(alt)
            except Exception as e2:
                logger.warning(f"Fallback model {alt} failed: {e2}")
        raise

async def get_cached_data(cache_key: str, ttl: int = 300) -> Optional[Any]:
    """Get cached data from Redis with TTL check"""
    if not redis_client:
        return None
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            # Check if it's a pickled object or JSON string
            try:
                return pickle.loads(cached_data)
            except:
                return json.loads(cached_data.decode('utf-8'))
        return None
    except Exception as e:
        logger.warning(f"Cache retrieval error for key {cache_key}: {e}")
        return None

async def set_cached_data(cache_key: str, data: Any, ttl: int = 300) -> None:
    """Set cached data in Redis with TTL"""
    if not redis_client:
        return
    
    try:
        # Try to pickle complex objects, fallback to JSON for simple ones
        try:
            serialized_data = pickle.dumps(data)
        except:
            serialized_data = json.dumps(data).encode('utf-8')
        
        redis_client.setex(cache_key, ttl, serialized_data)
        logger.debug(f"Cached data for key {cache_key} with TTL {ttl}s")
    except Exception as e:
        logger.warning(f"Cache storage error for key {cache_key}: {e}")

def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate a consistent cache key from parameters"""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        if v is not None:
            # Create hash for image content to avoid huge keys
            if k == 'image_content' and isinstance(v, bytes):
                key_parts.append(f"{k}:{hashlib.md5(v).hexdigest()[:8]}")
            else:
                key_parts.append(f"{k}:{str(v)}")
    return ":".join(key_parts)

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

async def generate_ai_report_async(prompt: str, image_content: bytes, timeout: int = None) -> str:
    """Generate AI report asynchronously with timeout control"""
    # Use environment variable for production timeout, fallback to 5 seconds for development
    if timeout is None:
        timeout = int(os.getenv('AI_TIMEOUT', '5'))
    
    def _generate_report():
        try:
            # Use model with fallbacks
            model = get_gemini_model()
            response = model.generate_content([prompt, Image.open(io.BytesIO(image_content))])
            return response.text
        except Exception as e:
            logger.warning(f"Gemini API error: {str(e)}")
            raise
    
    try:
        # Run with timeout to prevent long waits
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(executor, _generate_report),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"AI report generation timed out after {timeout} seconds, using fallback")
        # Return a structured fallback report instead of empty string
        fallback_report = {
            "status": "timeout_fallback",
            "message": "AI service timeout - using cached template",
            "report": {
                "summary": "Report generation timed out. Please try again or contact support.",
                "recommendations": [
                    "Check your internet connection",
                    "Try uploading a smaller image",
                    "Contact support if the issue persists"
                ],
                "severity": "medium",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Cache the fallback report for future use
        cache_key = f"fallback_report_{hash(str(image_content[:100]))}"
        await set_cached_data(cache_key, fallback_report, CACHE_TTL['ai_report'])
        logger.info(f"Generated and cached fallback report due to timeout after {timeout}s")
        return json.dumps(fallback_report, indent=2)

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
    if cached_report:
        logger.info(f"Using cached AI report for issue {issue_id}")
        return cached_report

    # Build location string
    location_str = (
        f"{address}, {zip_code}" if address and address.lower() != "not specified" and zip_code
        else address if address and address.lower() != "not specified"
        else f"Coordinates: {latitude}, {longitude}" if latitude and longitude
        else "Unknown Location"
    )

    # Authority data with caching
    authority_data = await get_authority_data_cached(zip_code, address, issue_type, latitude, longitude, category)
    responsible_authorities = authority_data.get("responsible_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])
    available_authorities = authority_data.get("available_authorities", [{"name": "City Department", "email": "snapfix@momntumai.com", "type": "general"}])

    # Time and IDs
    timezone_str = await get_timezone_cached(latitude, longitude)
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    local_time = now.strftime("%Y-%m-%d %H:%M")
    utc_time = now.astimezone(pytz.UTC).strftime("%H:%M")
    report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
    report_id = f"SNAPFIX-{now.year}-{report_number}"
    image_filename = f"IMG1_{now.strftime('%Y%m%d_%H%M')}.jpg"
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"

    # Department mapping and normalization
    issue_department_map = await get_department_mapping_cached()
    normalized_issue_type = issue_type.lower().replace(" ", "_")
    department = None
    if normalized_issue_type in issue_department_map:
        try:
            dep_val = issue_department_map[normalized_issue_type]
            department = dep_val[0] if isinstance(dep_val, list) and dep_val else dep_val
        except Exception:
            department = None
    if not department:
        for key, val in issue_department_map.items():
            if normalized_issue_type in key or normalized_issue_type.replace("_", " ") in key:
                department = val[0] if isinstance(val, list) and val else val
                break

    decline_prompt = f"- Decline Reason: {decline_reason}\n" if decline_reason else ""
    zip_code_prompt = f"- Zip Code: {zip_code}\n" if zip_code else ""

    prompt = f"""
You are an AI assistant for SnapFix AI, generating infrastructure issue reports.
Analyze the input below and return a structured JSON report (no markdown, no explanation).
Input:
- Issue Type: {issue_type.title()}
- Severity: {severity}
- Confidence: {confidence:.1f}%
- Description: {description}
- Category: {category}
- Location: {location_str}
- Issue ID: {issue_id}
- Responsible Department: {department}
- Map Link: {map_link}
- Priority: {priority}
{decline_prompt}
{zip_code_prompt}

For recommended_actions, provide 2-3 specific, actionable steps with timeframes. Examples:
- Potholes: ["Fill pothole and mark with cones within 48 hours.", "Conduct follow-up inspection after repair."]
- Broken Streetlight: ["Schedule bulb replacement within 3 days.", "Check wiring and restore power."]
- Water Leakage: ["Inspect pipeline and stop leakage within 24 hours.", "Fix joints and test pressure."]

Return ONLY JSON with this schema:
{{
  "issue_overview": {{
    "type": "{issue_type.title()}",
    "severity": "{severity}",
    "summary_explanation": "Brief summary of the issue and impact.",
    "confidence": {confidence:.1f}
  }},
  "detailed_analysis": {{
    "root_causes": "Possible causes of the issue.",
    "potential_consequences_if_ignored": "Risks if the issue is not addressed.",
    "public_safety_risk": "low|medium|high",
    "environmental_impact": "low|medium|high|none",
    "structural_implications": "low|medium|high|none",
    "legal_or_regulatory_considerations": "Relevant regulations or null",
    "feedback": "User-provided decline reason: {decline_reason}" if decline_reason else null
  }},
  "recommended_actions": ["Action 1", "Action 2"],
  "responsible_authorities_or_parties": {json.dumps(responsible_authorities)},
  "available_authorities": {json.dumps(available_authorities)},
  "additional_notes": "Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}.",
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
    "zip_code": "{zip_code if zip_code else 'N/A'}",
    "address": "{address if address else 'Not specified'}"
  }}
}}
Keep the report under 200 words, professional, and specific to the issue type and description.
"""

    try:
        # Generate AI report with timeout and fallbacks
        ai_text = await generate_ai_report_async(prompt, image_content)
        logger.info(f"Gemini optimized report output: {ai_text[:200]}...")

        # Extract and validate JSON
        json_match = re.search(r'\{[\s\S]*\}', ai_text)
        if not json_match:
            raise ValueError("No valid JSON found in response")
        json_text = json_match.group(0)
        report = json.loads(json_text)

        # Validate report structure
        required_fields = [
            "issue_overview",
            "detailed_analysis",
            "recommended_actions",
            "responsible_authorities_or_parties",
            "available_authorities",
            "additional_notes",
            "template_fields"
        ]
        missing_fields = [field for field in required_fields if field not in report]
        if missing_fields:
            raise ValueError(f"Missing required fields in report: {missing_fields}")

        # Update report fields
        report["additional_notes"] = f"Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}."
        report["template_fields"]["map_link"] = map_link
        report["template_fields"]["zip_code"] = zip_code if zip_code else "N/A"
        report["template_fields"]["address"] = address if address else "Not specified"
        report["responsible_authorities_or_parties"] = responsible_authorities
        report["available_authorities"] = available_authorities

        # Cache the generated report
        await set_cached_data(report_cache_key, report, CACHE_TTL['ai_report'])
        logger.info(f"Optimized report generated and cached for issue {issue_id}")
        return report
    except Exception as e:
        # Fallback structured report
        logger.warning(f"Attempt to generate optimized report failed: {str(e)}")
        actions = (
            [
                "Inspect site within 48 hours.",
                "Schedule repair and cordon area for safety."
            ] if severity.lower() != "low" else [
                "Schedule maintenance within 7 days.",
                "Monitor area for changes."
            ]
        )
        fallback_report = {
            "issue_overview": {
                "type": issue_type.title(),
                "category": category.title(),
                "severity": severity,
                "summary_explanation": f"Issue reported at {location_str}. {decline_reason or ''}".strip(),
                "confidence": confidence
            },
            "detailed_analysis": {
                "root_causes": "Possible causes of the issue.",
                "potential_consequences_if_ignored": "Risks if the issue is not addressed.",
                "public_safety_risk": "medium",
                "environmental_impact": "none",
                "structural_implications": "low",
                "legal_or_regulatory_considerations": "Refer to local regulations.",
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
                "zip_code": zip_code if zip_code else "N/A",
                "address": address if address else "Not specified"
            }
        }
        if decline_reason:
            fallback_report["issue_overview"]["summary_explanation"] += f" Declined due to: {decline_reason}."
        await set_cached_data(report_cache_key, fallback_report, CACHE_TTL['ai_report'])
        logger.info(f"Fallback optimized report cached for issue {issue_id}")
        return fallback_report