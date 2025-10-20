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
            model = genai.GenerativeModel("gemini-1.5-flash")
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
    
    # Check cache for similar reports first (enhanced caching strategy)
    cache_key = generate_cache_key(
        "ai_report_v2",
        issue_type=issue_type,
        severity=severity,
        category=category,
        image_content=image_content,
        zip_code=zip_code,
        address_hash=hashlib.md5(address.encode()).hexdigest()[:8]
    )
    
    cached_report = await get_cached_data(cache_key, CACHE_TTL['ai_report'])
    if cached_report:
        logger.info(f"Using cached report for similar issue {issue_id}")
        # Update timestamps and IDs for the cached report
        cached_report["report_id"] = report_id
        cached_report["issue_id"] = issue_id
        cached_report["timestamp"] = local_time
        cached_report["utc_time"] = utc_time
        cached_report["template_fields"]["oid"] = report_id
        cached_report["template_fields"]["timestamp"] = local_time
        cached_report["template_fields"]["utc_time"] = utc_time
        cached_report["template_fields"]["tracking_link"] = f"https://momentum-ai.org/track/{report_id}"
        return cached_report
    prompt = f"""
    Analyze this infrastructure issue image and generate a detailed report in JSON format.
    
    Issue Details:
    - Type: {issue_type}
    - Description: {description}
    - Severity: {severity}
    - Location: {address}
    - Category: {category}
    - Priority: {priority}
    - Confidence: {confidence}%
    
    Generate a JSON response with these exact fields:
    {{
        "issue_summary": "Brief summary of the issue",
        "detailed_analysis": "Detailed technical analysis",
        "recommended_action": "Specific action needed",
        "estimated_cost": "Cost estimate in USD",
        "timeline": "Expected resolution time",
        "safety_concerns": "Any safety issues",
        "materials_needed": "Required materials/equipment",
        "priority_justification": "Why this priority level"
    }}
    
    Keep responses concise but informative. Focus on actionable insights.
    """
    
    # Try AI generation with fast timeout and immediate fallback
    ai_success = False
    for attempt in range(1):  # Only 1 attempt for speed
        try:
            # Generate AI report asynchronously with 3-second timeout
            response_text = await generate_ai_report_async(prompt, image_content, timeout=3)
            logger.info(f"AI report generated quickly (attempt {attempt + 1}) for issue {issue_id}")
            
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
            
            ai_success = True
            logger.info(f"Optimized AI report generated successfully for issue {issue_id}")
            return report
            
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"AI generation timeout (attempt {attempt + 1}): {str(e)} - using fast fallback")
            break
        except Exception as e:
            logger.warning(f"AI generation attempt {attempt + 1} failed: {str(e)} - using fast fallback")
            break
    
    # FAST FALLBACK: Immediate response without AI delay
    logger.info(f"Using fast fallback report for issue {issue_id} - responding immediately")
    
    # Issue-specific action templates for quick response
    action_templates = {
        "pothole": {
            "issue_summary": f"Pothole detected on {address} requiring immediate attention",
            "detailed_analysis": f"Road surface damage identified with {severity.lower()} severity level. Pothole poses safety risk to vehicles and pedestrians.",
            "recommended_action": "Fill pothole with appropriate asphalt mixture and compact surface",
            "estimated_cost": "$150-300 USD",
            "timeline": "2-5 business days",
            "safety_concerns": "Vehicle damage risk, pedestrian tripping hazard",
            "materials_needed": "Asphalt mix, compaction equipment, safety cones",
            "priority_justification": f"Classified as {priority.lower()} priority due to safety implications"
        },
        "streetlight": {
            "issue_summary": f"Street lighting issue reported at {address}",
            "detailed_analysis": f"Lighting infrastructure problem with {severity.lower()} impact on public safety and visibility.",
            "recommended_action": "Inspect electrical connections and replace faulty components",
            "estimated_cost": "$75-200 USD",
            "timeline": "1-3 business days",
            "safety_concerns": "Reduced visibility, increased crime risk",
            "materials_needed": "LED bulbs, electrical components, safety equipment",
            "priority_justification": f"Set as {priority.lower()} priority for public safety"
        },
        "drainage": {
            "issue_summary": f"Drainage system issue identified at {address}",
            "detailed_analysis": f"Water management problem with {severity.lower()} severity affecting local infrastructure.",
            "recommended_action": "Clear blockages and inspect drainage system integrity",
            "estimated_cost": "$200-500 USD",
            "timeline": "3-7 business days",
            "safety_concerns": "Flooding risk, structural damage potential",
            "materials_needed": "Drainage equipment, cleaning tools, inspection cameras",
            "priority_justification": f"Assigned {priority.lower()} priority based on flood risk assessment"
        },
        "sidewalk": {
            "issue_summary": f"Sidewalk maintenance required at {address}",
            "detailed_analysis": f"Pedestrian pathway issue with {severity.lower()} severity impacting accessibility.",
            "recommended_action": "Repair concrete surface and ensure ADA compliance",
            "estimated_cost": "$300-800 USD",
            "timeline": "5-10 business days",
            "safety_concerns": "Pedestrian safety, accessibility compliance",
            "materials_needed": "Concrete mix, leveling tools, safety barriers",
            "priority_justification": f"Rated {priority.lower()} priority for pedestrian safety"
        }
    }
    
    # Select appropriate template or use generic
    template_key = next((key for key in action_templates.keys() if key in issue_type.lower()), "generic")
    if template_key == "generic":
        fallback_report_data = {
            "issue_summary": f"{issue_type} issue reported at {address}",
            "detailed_analysis": f"Infrastructure issue with {severity.lower()} severity requiring attention. Issue category: {category}.",
            "recommended_action": f"Inspect and address {issue_type.lower()} according to municipal standards",
            "estimated_cost": "$100-500 USD",
            "timeline": "3-7 business days",
            "safety_concerns": "Standard safety protocols apply",
            "materials_needed": "Standard maintenance equipment and materials",
            "priority_justification": f"Classified as {priority.lower()} priority based on severity assessment"
        }
    else:
        fallback_report_data = action_templates[template_key]
    
    actions = [fallback_report_data["recommended_action"], "Conduct professional inspection"]
    
    # Update fallback report with structured data
    fallback_report = {
        "issue_overview": {
            "issue_type": issue_type.title(),
            "severity": severity.lower(),
            "confidence": confidence,
            "category": category,
            "summary_explanation": fallback_report_data["issue_summary"]
        },
        "detailed_analysis": {
            "root_causes": fallback_report_data["detailed_analysis"],
            "potential_consequences_if_ignored": fallback_report_data["safety_concerns"],
            "public_safety_risk": severity.lower(),
            "environmental_impact": "low",
            "structural_implications": "medium" if issue_type in ["structural_damage", "property_damage"] else "low",
            "legal_or_regulatory_considerations": "Local regulations may apply",
            "feedback": f"User-provided decline reason: {decline_reason}" if decline_reason else None,
            "estimated_cost": fallback_report_data["estimated_cost"],
            "materials_needed": fallback_report_data["materials_needed"],
            "priority_justification": fallback_report_data["priority_justification"]
        },
        "recommended_actions": [fallback_report_data["recommended_action"], "Conduct professional inspection"],
        "responsible_authorities_or_parties": responsible_authorities,
        "available_authorities": available_authorities,
        "timeline": fallback_report_data["timeline"],
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
            "generation_method": "Fast Template Response",
            "response_time": "< 1 second"
        }
    }
    
    if decline_reason:
        fallback_report["issue_overview"]["summary_explanation"] += f" Declined due to: {decline_reason}."
    
    # Cache the generated report for future similar requests
    await set_cached_data(cache_key, fallback_report, CACHE_TTL['ai_report'])
    logger.info(f"Fallback report generated and cached for issue {issue_id}")
    
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