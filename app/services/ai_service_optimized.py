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
from urllib.parse import urlparse
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from services.report_summary_service import ReportSummaryBuilder

# Redis connection for caching (ENV-aware, REDIS_URL respected)
try:
    env = os.getenv("ENV", "development").lower()
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        parsed = urlparse(redis_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        password = parsed.password
        db = int((parsed.path or "/0").lstrip("/") or 0)
        redis_client = redis.Redis(host=host, port=port, password=password, db=db, decode_responses=False)
    else:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        db = int(os.getenv("REDIS_DB", "0"))
        redis_client = redis.Redis(host=host, port=port, password=password, db=db, decode_responses=False)

    # Attempt ping; reduce noise in dev
    redis_client.ping()
    logger.info(f"Redis connected for AI service caching ({host}:{port}, db={db})")
except Exception as e:
    redis_client = None
    log_fn = logger.warning if env == "production" else logger.info
    log_fn(f"Redis not available for AI service (env={os.getenv('ENV','development')}): {e}")

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
_MODEL = None

def get_gemini_model():
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        _MODEL = genai.GenerativeModel(model_name)
        return _MODEL
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
                _MODEL = genai.GenerativeModel(alt)
                return _MODEL
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
        timeout = int(os.getenv('AI_TIMEOUT', '8'))
    
    def _generate_report():
        try:
            # Use model with fallbacks
            model = get_gemini_model()
            # Open image in a context manager to avoid resource leaks
            with Image.open(io.BytesIO(image_content)) as img:
                response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            # Log and propagate to trigger fallback upstream
            logger.warning(f"Gemini API error: {str(e)}")
            raise
    
    try:
        # Run with timeout to prevent long waits
        # Use asyncio.to_thread for clarity and proper cooperative scheduling
        return await asyncio.wait_for(asyncio.to_thread(_generate_report), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"AI report generation timed out after {timeout} seconds, using fallback")
        logger.info("⚠️ Gemini response delayed — falling back gracefully instead of freezing")

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

    # NEW: define runtime fields and identifiers used later
    timezone_str = await get_timezone_cached(latitude, longitude)
    timezone = pytz.timezone(timezone_str or "UTC")
    now = datetime.now(timezone)
    local_time = now.strftime("%Y-%m-%d %H:%M")
    utc_time = now.astimezone(pytz.UTC).strftime("%H:%M")
    report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
    report_id = f"eaiser-{now.year}-{report_number}"
    image_filename = f"IMG1_{now.strftime('%Y%m%d_%H%M')}.jpg"
    map_link = (
        f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    )

    # Enrich address for UI if original address is unknown
    addr_clean = (address or "").strip()
    addr_unknown = addr_clean.lower() in {"unknown address", "not specified", "", "n/a"}
    display_address = addr_clean if not addr_unknown else (
        (f"Zip {zip_code}" if zip_code else (f"Coordinates: {latitude}, {longitude}" if latitude and longitude else "Unknown Location"))
    )
    # Attempt to enrich display_address with city/state using zip authorities
    try:
        zip_authorities = await load_json_data("zip_code_authorities.json")
        if zip_code and isinstance(zip_authorities, dict):
            z = zip_authorities.get(str(zip_code)) or zip_authorities.get(zip_code)
            if isinstance(z, dict) and addr_unknown:
                # First try explicit city/state keys
                city = z.get("city") or z.get("City")
                state = z.get("state") or z.get("State")
                if city and state:
                    display_address = f"{city}, {state}"
                else:
                    # Derive city prefix from any authority 'name' like 'Fairview City Department'
                    # Prefer 'general', fallback to first available list
                    auth_lists = []
                    if "general" in z and isinstance(z["general"], list) and z["general"]:
                        auth_lists = z["general"]
                    else:
                        for v in z.values():
                            if isinstance(v, list) and v:
                                auth_lists = v
                                break
                    if auth_lists:
                        name = auth_lists[0].get("name", "")
                        m = re.match(r"^([A-Za-z][A-Za-z\s'-]+?)(?:\s+(City|Public|Sanitation|Police|Emergency|Fire|Building|Animal|Environmental|Water|Code|Transportation)\b|\s+Department\b|\s+Works\b)", name)
                        city_guess = (m.group(1).strip() if m else name.split()[0]).strip()
                        if city_guess:
                            display_address = city_guess
    except Exception:
        pass

    # Check if similar report exists in cache and refresh dynamic fields
    cached_report = await get_cached_data(report_cache_key, CACHE_TTL['ai_report'])
    if cached_report:
        logger.info(f"Using cached AI report for issue {issue_id} (refreshing runtime fields)")
        tf = cached_report.get("template_fields", {})
        tf["oid"] = report_id
        tf["timestamp"] = local_time
        tf["utc_time"] = utc_time
        tf["map_link"] = map_link
        tf["zip_code"] = zip_code if zip_code else "N/A"
        tf["address"] = display_address
        tf["image_filename"] = tf.get("image_filename") or image_filename
        cached_report["template_fields"] = tf

        # Ensure aliases for UI
        overview = cached_report.get("issue_overview", {})
        if overview:
            overview["type"] = overview.get("type") or overview.get("issue_type") or (issue_type.title() if issue_type else "Issue")
            expl = (overview.get("summary_explanation") or "").strip()
            overview["summary"] = (expl.split("\n")[0].strip() if "\n" in expl else expl) or f"{overview['type']} reported at {display_address}."
            cached_report["issue_overview"] = overview

        detailed = cached_report.get("detailed_analysis", {})
        if detailed and "potential_consequences_if_ignored" in detailed:
            detailed["potential_impact"] = detailed["potential_consequences_if_ignored"]
            cached_report["detailed_analysis"] = detailed

        # Refresh the address in additional notes, if present
        if isinstance(cached_report.get("additional_notes"), str):
            cached_report["additional_notes"] = re.sub(r"Location: .*?\. ", f"Location: {display_address}. ", cached_report["additional_notes"]) 

        return cached_report

    # Build location string
    location_str = (
        f"{display_address}. Zip: {zip_code}" if zip_code else display_address
    )

    # Authority data with caching (fetch in parallel later)
    authority_data = None
    responsible_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
    available_authorities = [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]

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

    # Load valid issue types dynamically for the prompt
    issue_category_map = await load_json_data("issue_category_map.json")
    valid_issue_types_list = list(issue_category_map.keys())
    # Format for prompt (comma separated, Title Case)
    valid_issue_types_str = ", ".join([t.replace("_", " ").title() for t in valid_issue_types_list])

    prompt = f"""
You are an AI assistant for eaiser AI, generating infrastructure issue reports.
Analyze the input below and return a structured JSON report (no markdown, no explanation).

Valid Issue Types:
{valid_issue_types_str}

CRITICAL INSTRUCTION ON REALISM & SAFETY:
1. FAKE/IRRELEVANT IMAGES:
   - If the image is a CARTOON, ANIMATION, DRAWING, VIDEO GAME SCREENSHOT, or OBVIOUSLY FAKE/GENERATED:
     - Set "issue_detected" to false.
     - Set "detected_issue_type" to "None".
     - Set "ai_confidence_percent" to 5.
     - In "image_analysis", explicitly state: "Image appears to be a cartoon, animation, or simulation, not a real-world infrastructure issue."
   - If the image is unrelated to infrastructure (e.g., a selfie, food, indoor furniture):
     - Set "issue_detected" to false.

2. FIRES - STRICT CLASSIFICATION:
   - Distinguish between "Dangerous Fire" (Wildfire, Building Fire, Out of Control) and "Controlled Fire" (Campfire, Bonfire, BBQ, Fire Pit, Religious Ceremony, Burning Leaves, Festival).
   - If the image shows a SAFE, CONTROLLED fire (e.g., inside a fire pit, people sitting around, BBQ grill, small religious fire, bonfire, burning wood pile, large crowd watching fire, festival celebration):
     - Classify the Issue Type as "Other" (do NOT use "Fire" or "Wildfire").
     - Set "issue_detected" to false.
     - Set "ai_confidence_percent" to 40 (Must be < 50%).
     - In "image_analysis", explicitly state: "Detected controlled fire/bonfire; not a public safety threat."
     - In "summary_explanation", use words like "bonfire", "controlled fire", "festival", or "ceremony".
   - ONLY classify as "Fire" if it poses a threat to public safety (e.g., house on fire, forest burning uncontrolled).
   - IMPORTANT: If you see people standing calmly near the fire, it is likely a BONFIRE/FESTIVAL. Classify as "Other" (40% confidence).

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

Return ONLY JSON with actual values filled in (do NOT use template variables like {{Issue_Type}} - use the actual values I provided above):
{{
  "issue_overview": {{
    "type": "{issue_type.title()}",
    "severity": "{severity}",
    "summary_explanation": "Our AI detected a {issue_type} in {location_str}. The image shows {description}. Based on the location and context, this incident has been classified as {priority} due to {category}. Report ID: {report_id}.",
    "confidence": {confidence:.1f}
  }},
  "ai_evaluation": {{
    "image_analysis": "Describe what is happening in the image and which visual cues support your conclusion.",
    "issue_detected": true|false,
    "detected_issue_type": "One of the Valid Issue Types listed above or 'None'",
    "ai_confidence_percent": 0,
    "rationale": "Brief justification referencing image clarity and visual evidence. Use integer for ai_confidence_percent only (no %). If issue_detected is false, set ai_confidence_percent between 0 and 10; if true, set between 10 and 100 based on clarity and understanding."
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
    "address": "{address if address else 'Not specified'}",
    "confidence": {confidence:.1f}
  }}
}}
Keep the report under 200 words, professional, and specific to the issue type and description.
"""

    try:
        # Generate AI report with timeout and fetch authorities concurrently
        authority_task = asyncio.create_task(get_authority_data_cached(zip_code, address, issue_type, latitude, longitude, category))
        ai_text = await generate_ai_report_async(prompt, image_content)
        logger.info(f"Gemini optimized report output: {ai_text[:200]}...")

        # Extract and validate JSON
        json_match = re.search(r'\{[\s\S]*\}', ai_text)
        if not json_match:
            raise ValueError("No valid JSON found in response")
        json_text = json_match.group(0)
        report = json.loads(json_text)

        # Helper to ensure required fields exist with sane defaults
        def _ensure_required_fields(rep: Dict[str, Any]) -> Dict[str, Any]:
            """Soft-fill missing required fields to avoid brittle failures.

            Adds keys with minimal, valid defaults so downstream consumers
            do not choke on missing fields when AI output is partial.
            """
            rep.setdefault("issue_overview", {
                "type": issue_type.title() if issue_type else "Issue",
                "category": category.title(),
                "severity": severity,
                "summary_explanation": description or "",
                "confidence": int(round(confidence))
            })
            rep.setdefault("detailed_analysis", {
                "root_causes": "Undetermined; requires inspection.",
                "potential_consequences_if_ignored": "Potential safety or compliance risks.",
                "public_safety_risk": severity.lower(),
                "environmental_impact": "low",
                "structural_implications": "low",
                "legal_or_regulatory_considerations": "Local regulations may apply.",
                "feedback": f"User-provided decline reason: {decline_reason}" if decline_reason else None
            })
            rep.setdefault("recommended_actions", [
                "Inspect site",
                "Schedule maintenance"
            ])
            rep.setdefault("responsible_authorities_or_parties", responsible_authorities)
            rep.setdefault("available_authorities", available_authorities)
            rep.setdefault("additional_notes", f"Location: {location_str}. View: {map_link}. Issue ID: {issue_id}. Zip: {zip_code or 'N/A'}.")
            tf = rep.setdefault("template_fields", {})
            tf.setdefault("oid", report_id)
            tf.setdefault("timestamp", local_time)
            tf.setdefault("utc_time", utc_time)
            tf.setdefault("priority", priority)
            tf.setdefault("tracking_link", f"https://momentum-ai.org/track/{report_id}")
            tf.setdefault("image_filename", image_filename)
            tf.setdefault("ai_tag", (issue_type or "Issue").title())
            tf.setdefault("app_version", "1.5.3")
            tf.setdefault("device_type", "Mobile (Generic)")
            tf.setdefault("map_link", map_link)
            tf.setdefault("zip_code", zip_code or "N/A")
            tf.setdefault("address", display_address)
            tf.setdefault("confidence", int(round(confidence)))
            rep["template_fields"] = tf
            return rep

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
            logger.info(f"Soft-filling missing fields in AI report: {missing_fields}")
            report = _ensure_required_fields(report)
            # Re-check after fill; only fail if still missing critical structure
            missing_fields = [field for field in required_fields if field not in report]
            if missing_fields:
                raise ValueError(f"Missing required fields in report after fill: {missing_fields}")

        # Update report fields
        report["additional_notes"] = f"Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}."
        report["template_fields"]["map_link"] = map_link
        report["template_fields"]["zip_code"] = zip_code if zip_code else "N/A"
        report["template_fields"]["address"] = display_address
        try:
            ov_conf = report.get("issue_overview", {}).get("confidence")
            report["template_fields"]["confidence"] = int(round(float(ov_conf if ov_conf is not None else confidence)))
        except Exception:
            report["template_fields"]["confidence"] = int(round(confidence))
        try:
            if authority_data is None:
                authority_data = await authority_task
            responsible_authorities = authority_data.get("responsible_authorities", responsible_authorities)
            available_authorities = authority_data.get("available_authorities", available_authorities)
        except Exception:
            pass
        report["responsible_authorities_or_parties"] = responsible_authorities
        report["available_authorities"] = available_authorities

        # Ensure UI-friendly aliases for primary path and enforce minimum 6-line summary_explanation
        overview = report.get("issue_overview") or {}
        if overview:
            overview["type"] = overview.get("type") or overview.get("issue_type") or (issue_type.title() if issue_type else "Issue")
            ai_eval = report.get("ai_evaluation") or {}
            detected_type = (ai_eval.get("detected_issue_type") or "").strip()
            if detected_type and detected_type.lower() != "none":
                overview["type"] = detected_type
            expl = (overview.get("summary_explanation") or "").strip()
            ai_analysis = (ai_eval.get("image_analysis") or "").strip()
            if ai_analysis and not expl.startswith("AI Analysis:"):
                expl = f"AI Analysis: {ai_analysis}\n{expl}".strip()
            lines = [l for l in expl.split("\n") if l.strip()]
            ai_conf = ai_eval.get("ai_confidence_percent")
            try:
                ai_conf_val = int(round(float(ai_conf))) if ai_conf is not None else None
            except Exception:
                ai_conf_val = None
            if ai_conf_val is not None:
                overview["confidence"] = ai_conf_val
            if len(lines) < 6:
                extras = [
                    f"Location context: {location_str}.",
                    f"Issue type: {overview.get('type')}, severity: {severity.lower()}, confidence: {overview.get('confidence', confidence):.0f}%.",
                    f"Category: {category}.",
                    f"Potential impact: {report.get('detailed_analysis', {}).get('potential_consequences_if_ignored', 'N/A')}.",
                    f"Initial action: {', '.join(report.get('recommended_actions', [])[:2]) or 'N/A'}.",
                    f"Tracking reference: {report.get('template_fields', {}).get('oid', '')}."
                ]
                for x in extras:
                    if len(lines) >= 6:
                        break
                    lines.append(x)
                overview["summary_explanation"] = "\n".join(lines)
                expl = overview["summary_explanation"]
            overview["summary"] = (expl.split("\n")[0].strip() if "\n" in expl else expl) or f"{overview['type']} reported at {display_address}."
            report["issue_overview"] = overview
            # Normalize ai_evaluation fields with sensible defaults
            if ai_eval:
                ai_eval["issue_detected"] = bool(ai_eval.get("issue_detected")) if ai_eval.get("issue_detected") is not None else (overview.get("type") not in (None, "", "None"))
                if ai_conf_val is None:
                    try:
                        ai_eval["ai_confidence_percent"] = int(round(float(overview.get("confidence", confidence))))
                    except Exception:
                        ai_eval["ai_confidence_percent"] = int(round(confidence))
                else:
                    ai_eval["ai_confidence_percent"] = ai_conf_val
                if not detected_type:
                    ai_eval["detected_issue_type"] = overview.get("type")
                if not ai_analysis:
                    ai_eval["image_analysis"] = "No explicit image analysis provided."
                ai_eval["rationale"] = ai_eval.get("rationale") or "Rationale not provided."
                report["ai_evaluation"] = ai_eval
        detailed = report.get("detailed_analysis") or {}
        if detailed and "potential_consequences_if_ignored" in detailed:
            detailed["potential_impact"] = detailed["potential_consequences_if_ignored"]
            report["detailed_analysis"] = detailed

        # Post-adjust confidence for benign/minor scenes to reduce false positives
        try:
            overview = report.get("issue_overview", {})
            ai_eval = report.get("ai_evaluation", {})
            desc_text = str(overview.get("summary_explanation", "")).lower()
            eval_text = str(ai_eval.get("image_analysis", "")).lower()
            combined = f"{desc_text} {eval_text}"
            severity_val = str(overview.get("severity", "medium")).lower()
            conf_val = overview.get("confidence")
            try:
                conf_val = int(round(float(conf_val))) if conf_val is not None else None
            except Exception:
                conf_val = None
            danger_words = ["out of control","emergency","uncontrolled","explosion","collapse","wildfire","house fire","building fire","structure fire","spreading wildly","sirens"]
            controlled_fire = [
                "campfire","bonfire","bon fire","bornfire","bbq","barbecue","barbeque","grill","fire pit","controlled burn","festival",
                "celebration","diwali","diya","candle","incense","lamp","stove","kitchen","smoke machine","stage","fireplace","hearth",
                "wood stove","fire ring","camp fire","marshmallow","roasting","gathering","chairs around","burning wood","logs burning",
                "small fire","contained fire","cooking","recreational","holika","dahan","lohri","ceremony","ritual","crowd","people watching",
                "people standing","spectators","religious","tradition","custom"
            ]
            minor_words = ["minor","small","tiny","cosmetic","scratch","smudge","dust","stain","normal","benign"]
            fake_words = ["cartoon", "animation", "drawing", "illustration", "game screenshot", "video game", "simulated", "generated", "fake", "render", "clipart", "sketch", "painting"]

            has_danger = any(w in combined for w in danger_words)
            is_controlled = any(w in combined for w in controlled_fire)
            is_minor = any(w in combined for w in minor_words)
            is_fake = any(w in combined for w in fake_words)
            
            animal_tokens = ["animal","deer","boar","hog","dog","cow","cat","wildlife","goat","pig","carcass","roadkill"]
            accident_tokens = ["accident","collision","crash","hit","struck","run over","under car","under vehicle"]
            people_tokens = ["people", "crowd", "spectators", "gathering", "watching", "standing", "men", "women", "children", "group"]
            
            has_animal = any(w in combined for w in animal_tokens)
            has_accident = any(w in combined for w in accident_tokens)
            has_people = any(w in combined for w in people_tokens)
            issue_detected = bool(ai_eval.get("issue_detected"))
            
            # Special check for Fire + People -> likely controlled
            if (overview.get("type") or "").lower() == "fire" and has_people and not any(w in combined for w in ["house", "building", "structure", "vehicle", "car"]):
                 is_controlled = True

            if is_fake:
                new_conf = 5
                issue_detected = False
                overview["type"] = "None"
                ai_eval["detected_issue_type"] = "None"
                ai_eval["image_analysis"] = "Detected as fake/cartoon/animated image."
                ai_eval["rationale"] = "Image classified as non-realistic or simulated."
            elif not issue_detected:
                hazard_tokens = ["hazard","danger","out of control","emergency","injury","uncontrolled","explosion","collapse","severe","major","wildfire","accident","collision","leak","burst"]
                has_hazard = any(w in combined for w in hazard_tokens)
                new_conf = 40 if not has_hazard else 80
                issue_detected = bool(has_hazard)
            elif is_controlled and not has_danger:
                new_conf = 40  # Explicitly set to 40% for bonfires/controlled fires
                issue_detected = False
                overview["type"] = "Other"
                ia = ai_eval.get("image_analysis") or ""
                if "no public issue" not in ia.lower():
                    ai_eval["image_analysis"] = "Detected controlled fire/bonfire; not a public safety threat."
            elif is_minor and not has_danger:
                new_conf = max(75, int(conf_val if conf_val is not None else (ai_eval.get("ai_confidence_percent") or 70)))
            elif has_animal and has_accident and not any(w in combined for w in ["fire","flame","burning","smoke"]):
                overview["type"] = "animal_accident"
                new_conf = max(85, int(conf_val if conf_val is not None else (ai_eval.get("ai_confidence_percent") or 80)))
                issue_detected = True
            else:
                # Prefer fallen tree classification if strong cues present
                tree_tokens = [
                    "fallen tree","tree fallen","downed tree","tree down","uprooted",
                    "tree across","tree blocking road","tree blocking the road","tree on road",
                    "branches","branch","trunk","logs","log","limb","limbs",
                    "branches across","debris from tree","fallen branches","tree debris"
                ]
                if any(w in combined for w in tree_tokens):
                    overview["type"] = "tree_fallen"
                    new_conf = max(85, int(conf_val if conf_val is not None else (ai_eval.get("ai_confidence_percent") or 80)))
                    issue_detected = True
            if 'new_conf' not in locals():
                new_conf = conf_val if conf_val is not None else ai_eval.get("ai_confidence_percent") or 60
            new_conf = int(max(0, min(100, new_conf)))
            # Use dynamically loaded valid types + standard fallbacks
            allowed_types = set(valid_issue_types_list) | {"unknown", "other"}
            ttype = (overview.get("type") or "").lower()
            if ttype and ttype not in allowed_types:
                new_conf = min(new_conf, 40)
                ai_eval["issue_detected"] = False
                ia = ai_eval.get("image_analysis") or ""
                if "no public issue" not in ia.lower():
                    ai_eval["image_analysis"] = "I did not find any public issue."
            # Clamp confidence for non-specific types
            if ttype in ("other", "unknown", "none"):
                new_conf = min(new_conf, 50)
            # Ensure target issue types have >=75 confidence unless controlled benign
            try:
                ttype = (overview.get("type") or "").lower()
                target_types = [
                    "pothole", "road damage", "road_damage",
                    "fire", "wildfire",
                    "dead animal", "animal carcass", "animal_carcass",
                    "garbage", "trash", "waste",
                    "flood", "waterlogging",
                    "tree fallen", "fallen tree", "tree_fallen",
                    "public toilet", "public_toilet_issue"
                ]
                is_target = any(tt in ttype for tt in target_types)
                if is_target and not (is_controlled and not has_danger):
                    new_conf = max(new_conf, 75)
                elif is_target and is_controlled and not has_danger:
                    # Explicitly downgrade controlled fires even if they are target types
                    new_conf = min(new_conf, 40)
                    ai_eval["issue_detected"] = False
                    overview["type"] = "Other"
                    if "no public issue" not in (ai_eval.get("image_analysis") or "").lower():
                         ai_eval["image_analysis"] = f"Detected controlled fire ({overview.get('type')}); not a public safety issue."
            except Exception:
                pass
            
            # Force "other" issues to have low confidence (< 20%)
            if (overview.get("type") or "").lower() == "other":
                new_conf = min(new_conf, 15)
            
            # Additional safety: If user explicitly mentions bonfire/controlled in issue type/description, force low confidence
            # This handles cases where AI misses it but user context provided it, or simple keyword match
            user_input_text = (description + " " + issue_type).lower()
            if any(w in user_input_text for w in controlled_fire) and not any(w in user_input_text for w in danger_words):
                 new_conf = min(new_conf, 40)
                 ai_eval["issue_detected"] = False
                 overview["type"] = "Other"
                
            overview["confidence"] = new_conf
            ai_eval["ai_confidence_percent"] = new_conf
            ai_eval["issue_detected"] = issue_detected
            report["issue_overview"] = overview
            report["ai_evaluation"] = ai_eval
        except Exception:
            pass

        # RE-FETCH AUTHORITIES IF ISSUE TYPE CHANGED
        # This fixes the issue where authority assignment is wrong because it used the initial (potentially wrong) classification.
        final_issue_type = (report.get("issue_overview", {}).get("type") or issue_type).lower().replace(" ", "_")
        initial_issue_type = issue_type.lower().replace(" ", "_")
        
        if final_issue_type != initial_issue_type and final_issue_type in allowed_types:
            logger.info(f"Issue type changed from {initial_issue_type} to {final_issue_type}. Re-fetching authorities.")
            try:
                # Re-fetch authority data with the NEW issue type
                new_authority_data = await get_authority_data_cached(
                    zip_code, address, final_issue_type, latitude, longitude, category
                )
                
                # Update variables
                responsible_authorities = new_authority_data.get("responsible_authorities", responsible_authorities)
                available_authorities = new_authority_data.get("available_authorities", available_authorities)
                
                # Update Report JSON
                report["responsible_authorities_or_parties"] = responsible_authorities
                report["available_authorities"] = available_authorities
                
                # Also try to update department in template fields if possible, though it's less critical
            except Exception as e:
                logger.warning(f"Failed to re-fetch authorities for new type {final_issue_type}: {e}")

        # Build concise summary per strict template (City, State, ZIP)
        def _extract_city_state(addr: str) -> (str, str):
            """Try to split 'City, State' from address; fallback to Unknowns."""
            if not addr:
                return "Unknown City", "Unknown State"
            parts = [p.strip() for p in str(addr).split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]
            # If only one part, treat as city
            return parts[0] if parts else "Unknown City", "Unknown State"

        # Determine risk tags from detailed analysis signals
        da = report.get("detailed_analysis", {}) or {}
        risk_tags: list = []
        for key in ["public_safety_risk", "environmental_impact", "structural_implications"]:
            val = (da.get(key) or "").strip()
            if val and val.lower() not in {"none", "", "n/a"}:
                # Convert snake_case key to readable words
                risk_tags.append(key.replace("_", " ").title())

        city_val, state_val = _extract_city_state(display_address)
        short_desc = overview.get("summary") or description or "no clear visual description"

        summary_builder = ReportSummaryBuilder()
        summary_text = summary_builder.build({
            "Issue_Type": overview.get("type") or issue_type,
            "City": city_val,
            "State": state_val,
            "Zip_Code": zip_code or "N/A",
            "Short_Visual_Description": short_desc,
            "Priority_Label": priority,
            "Risk_Tags": risk_tags,
        })

        # Attach concise summary to report for downstream consumers
        report["summary"] = summary_text

        # Cache the generated report
        await set_cached_data(report_cache_key, report, CACHE_TTL['ai_report'])
        logger.info(f"Optimized report generated and cached for issue {issue_id}")

        # === EAiSER FORMATTED ALERT TEMPLATE (Concise SUMMARY enforced) ===
        # Calculate color for confidence score
        if confidence >= 80:
            conf_color = "#4ade80" # Green
        elif confidence >= 50:
            conf_color = "#facc15" # Yellow
        else:
            conf_color = "#ef4444" # Red

        formatted_alert = f"""
Subject: EAiSER Alert – {overview.get('type', issue_type).title()} (ID: {report_id})
🚨 EAiSER INFRASTRUCTURE ALERT 🚨
Detected by EAiSER Ai Automated System
________________________________________
🧾 SUMMARY
{summary_text}
Report ID: {report_id}
________________________________________
📍 LOCATION DETAILS
Field\tInformation
Address\t{display_address}
Zip Code\t{zip_code or 'N/A'}
Coordinates\t{latitude}, {longitude}
Map Link\t{map_link}
________________________________________
🧠 AI REPORT SUMMARY
Field\tDetails
Issue Type\t{issue_type.title()}
AI Confidence\t<span style="color: {conf_color}; font-weight: bold;">{confidence:.2f}%</span><br><div style="width: 100%; max-width: 250px; height: 8px; background-color: #e5e7eb; border-radius: 4px; margin-top: 4px; overflow: hidden;"><div style="width: {confidence:.0f}%; height: 100%; background-color: {conf_color};"></div></div>
Priority\t{priority}
Time Reported\t{local_time} {timezone_str}
Report ID\t{report_id}
Impact Summary\t{report.get('detailed_analysis', {}).get('potential_consequences_if_ignored', 'Not available')}
Location Context\t{report.get('issue_overview', {}).get('location_context', display_address)}
________________________________________
🖼️ PHOTO EVIDENCE
📎 File: {image_filename}
________________________________________
📬 CONTACT & FOLLOW-UP
Please review and address this issue as soon as feasible.
For questions or to confirm completion, contact:
📧 support@momntumai.com
🔗 View Report on Map: {map_link}
________________________________________
Automated report generated via EAiSER AI by MomntumAI
© 2025 MomntumAI | All Rights Reserved
"""
        # Add formatted alert into report dictionary
        report["formatted_report"] = formatted_alert

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
        is_tree_issue = "tree" in (issue_type or "").lower() or "fallen" in (description or "").lower()
        is_animal_issue = "animal" in (issue_type or "").lower() or "animal" in (description or "").lower()
        if is_tree_issue:
            actions = [
                "Dispatch public works to clear fallen tree within 24 hours.",
                "Inspect area for powerline damage; coordinate with utilities.",
                "Place safety cones and detour signs until cleared."
            ]
        elif is_animal_issue:
            actions = [
                "Dispatch animal control officer within 2 hours.",
                "Secure area and notify nearby residents.",
                "Capture/relocate animal; document incident and outcomes."
            ]
        # Build fallback ai_evaluation
        _detected_type = issue_type.title() if issue_type else "None"
        _issue_detected = bool(issue_type and issue_type.strip().lower() not in ("none", "unknown", ""))
        try:
            _base_conf = int(round(float(confidence)))
        except Exception:
            _base_conf = 50
        _ai_confidence_percent = max(10, min(100, _base_conf)) if _issue_detected else max(0, min(10, _base_conf))
        # Clamp confidence for Other/Unknown/None
        try:
            if str(_detected_type).strip().lower() in ("other", "unknown", "none"):
                _ai_confidence_percent = min(_ai_confidence_percent, 50)
        except Exception:
            pass
        _image_analysis = (
            f"{_detected_type if _issue_detected else 'No obvious issue detected'}; "
            f"severity {severity.lower()}, confidence {_ai_confidence_percent}% based on available visual indicators and metadata."
        )

        fallback_report = {
            "issue_overview": {
                "type": _detected_type,
                "category": category.title(),
                "severity": severity,
                "summary_explanation": (
                    f"AI Analysis: {_image_analysis}\n"
                    f"Issue reported at {location_str}.\n"
                    f"Map context: {map_link}.\n"
                    f"Zip code context: {zip_code if zip_code else 'N/A'}; coordinates: {latitude}, {longitude}.\n"
                    f"The issue type is {_detected_type}; severity assessed as {severity.lower()} with {_ai_confidence_percent}% confidence.\n"
                    f"Visual indicators and metadata support the classification and estimated impact.\n"
                    f"Initial actions recommended: {actions[0]}{(' ' + actions[1]) if len(actions) > 1 else ''}.\n"
                    f"Please review and escalate to the appropriate authorities for resolution."
                ),
                "confidence": _ai_confidence_percent
            },
            "ai_evaluation": {
                "image_analysis": _image_analysis,
                "issue_detected": _issue_detected,
                "detected_issue_type": _detected_type,
                "ai_confidence_percent": _ai_confidence_percent,
                "rationale": "Derived from supplied fields; image clarity unverifiable in fallback."
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
                "address": display_address,
                "confidence": _ai_confidence_percent
            }
        }
        if decline_reason:
            fallback_report["issue_overview"]["summary_explanation"] += f" Declined due to: {decline_reason}."
        # UI-friendly aliases and mapping
        try:
            expl = (fallback_report.get("issue_overview", {}).get("summary_explanation") or "").strip()
            fallback_report["issue_overview"]["summary"] = (expl.split("\n")[0].strip() if "\n" in expl else expl) or f"{issue_type.title()} reported at {display_address}."
            da = fallback_report.get("detailed_analysis", {})
            if "potential_consequences_if_ignored" in da:
                da["potential_impact"] = da["potential_consequences_if_ignored"]
                fallback_report["detailed_analysis"] = da
        except Exception:
            pass

        # Build a concise human-readable summary for fallback using the ReportSummaryBuilder
        # This ensures the SUMMARY section is consistent even when AI times out.
        try:
            # Derive risk tags in a lightweight way from available signals
            risk_tags_fb = []
            da_fb = fallback_report.get("detailed_analysis", {})
            public_risk = (da_fb.get("public_safety_risk") or "").lower()
            if public_risk in ("medium", "high"):
                risk_tags_fb.append("Public Safety Risk")
            if (severity or "").lower() == "high":
                risk_tags_fb.append("Severe Issue")
            if (priority or "").lower() in ("urgent", "high"):
                risk_tags_fb.append("High Priority")
            # Deduplicate and cap to 6 tags
            risk_tags_fb = list(dict.fromkeys(risk_tags_fb))[:6]

            # Compose a short visual description from user description if available
            short_desc_fb = (description or f"{issue_type.title() if issue_type else 'Issue'} observed.").strip()

            # Use ReportSummaryBuilder to render the concise template
            builder_fb = ReportSummaryBuilder()
            summary_text_fb = builder_fb.build({
                # Prefer detected type when available
                "issue_type": fallback_report.get("ai_evaluation", {}).get("detected_issue_type") or (issue_type.title() if issue_type else "Issue"),
                # City/State may be unavailable in fallback; keep None for clarity
                "city": None,
                "state": None,
                "zip_code": zip_code or None,
                "short_visual_description": short_desc_fb,
                "priority_label": priority or "N/A",
                "risk_tags": risk_tags_fb,
            })

            # Attach concise summary to fallback report for downstream consumers
            fallback_report["summary"] = summary_text_fb
        except Exception as _e:
            # If summary construction fails, omit but do not block fallback
            logger.warning(f"Failed to build concise fallback summary: {_e}")

        # === EAiSER FORMATTED ALERT TEMPLATE for fallback ===
        # Calculate color for fallback confidence
        if _ai_confidence_percent >= 80:
            fb_conf_color = "#4ade80"
        elif _ai_confidence_percent >= 50:
            fb_conf_color = "#facc15"
        else:
            fb_conf_color = "#ef4444"

        fb_conf_bar_html = f'<div style="width: 100%; max-width: 250px; height: 8px; background-color: #e5e7eb; border-radius: 4px; margin-top: 4px; overflow: hidden;"><div style="width: {_ai_confidence_percent}%; height: 100%; background-color: {fb_conf_color};"></div></div>'

        formatted_alert = f"""
Subject: EAiSER Alert – {issue_type.title() if issue_type else 'Issue'} (ID: {report_id})
🚨 EAiSER INFRASTRUCTURE ALERT 🚨
Detected by EAiSER Ai Automated System
________________________________________
🧾 SUMMARY
{fallback_report.get('summary', 'No summary available')}
Report ID: {report_id}
________________________________________
📍 LOCATION DETAILS
Field\tInformation
Address\t{display_address}
Zip Code\t{zip_code or 'N/A'}
Coordinates\t{latitude}, {longitude}
Map Link\t{map_link}
________________________________________
🧠 AI REPORT SUMMARY
Field\tDetails
Issue Type\t{issue_type.title() if issue_type else 'N/A'}
AI Confidence\t<span style="color: {fb_conf_color}; font-weight: bold;">{_ai_confidence_percent}%</span><br>{fb_conf_bar_html}
Priority\t{priority}
Time Reported\t{local_time} {timezone_str}
Report ID\t{report_id}
Impact Summary\t{fallback_report.get('detailed_analysis', {}).get('potential_consequences_if_ignored', 'Not available')}
Location Context\t{display_address}
________________________________________
🖼️ PHOTO EVIDENCE
📎 File: {image_filename}
________________________________________
📬 CONTACT & FOLLOW-UP
Please review and address this issue as soon as feasible.
For questions or to confirm completion, contact:
📧 support@momntumai.com
🔗 View Report on Map: {map_link}
________________________________________
Automated report generated via EAiSER AI by MomntumAI
© 2025 MomntumAI | All Rights Reserved
"""
        fallback_report["formatted_report"] = formatted_alert

        await set_cached_data(report_cache_key, fallback_report, CACHE_TTL['ai_report'])
        logger.info(f"Fallback optimized report cached for issue {issue_id}")
        return fallback_report
