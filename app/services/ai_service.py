import google.generativeai as genai
from PIL import Image
import pillow_heif  # 🟢 Ticket 1: HEIC Support
pillow_heif.register_heif_opener() # Register HEIC opener
import io
import json
import logging
import aiofiles
import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import re
from pathlib import Path
from utils.timezone import get_timezone_name
from utils.location import get_authority_by_zip_code, get_authority
from typing import Optional, Dict, Any
import asyncio
from services.ai_service_v3 import get_ai_service_v3
from utils.v4_tables import (
    reconcile_and_enrich, get_tier, get_priority_for_issue,
    is_emergency, get_emergency_advisory, TIER_MAP,
    EMERGENCY_ADVISORIES, ADVISORY_PRECEDENCE
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from root directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# 🚀 Key Pool: Use multi-key rotation instead of single GEMINI_API_KEY
from services.gemini_key_pool import get_key_pool
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Kept for heuristic fallback check
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set; heuristic fallback may be used.")

# 🚀 Model getter now uses key pool for automatic key rotation
def get_gemini_model():
    """Get a Gemini model from the key pool (auto-rotates API keys)."""
    pool = get_key_pool()
    model, slot = pool.get_model(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )
    return model, slot

# Cache for JSON data to reduce file I/O
_json_cache = {}

async def load_json_data(file_name: str) -> dict:
    """Load JSON data from a file asynchronously with caching."""
    if file_name in _json_cache:
        logger.debug(f"Returning cached JSON data for {file_name}")
        return _json_cache[file_name]
    try:
        file_path = Path(__file__).parent.parent / "data" / file_name
        async with aiofiles.open(file_path, "r") as file:
            content = await file.read()
            data = json.loads(content)
        _json_cache[file_name] = data
        logger.debug(f"Loaded and cached JSON data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {str(e)}")
        return {}

async def classify_issue(image_content: bytes, description: str) -> tuple[str, str, float, str, str]:
    """Classify an infrastructure issue based on image and description.
    
    V4 Enhancement: After AI classification, the V4 Reconciler enforces:
    - Tier-based priority (TIER_MAP overrides AI)
    - Emergency 911 detection for Tier 0 issues
    - Scenario determination (A-F)
    - Consistency checks
    """
    # EAiSER V3 Integration: Use V3 for higher precision and deterministic output
    try:
        v3_service = get_ai_service_v3()
        v3_result = await v3_service.analyze_single_image(image_content)
        
        if v3_result.get("success"):
            data = v3_result["data"]
            meta = data.get("report_meta", {})
            ordered = data.get("ordered_issue_list", [])
            
            # Extract key metrics for legacy compatibility
            primary_issue = "Other"
            primary_type = "Known"
            tier_or_severity = "Tier 2"
            
            if ordered:
                primary_issue = ordered[0].get("issue", "Other")
                primary_type = ordered[0].get("type", "Known")
                tier_or_severity = ordered[0].get("tier_or_severity", "Tier 2")
            
            # 🛡️ GUARD: If the AI specifically says 'No Visible Public Infrastructure Issue', block it.
            # This matches the Website's 'No Issue Detected' guard.
            issue_detected = True
            if primary_issue == "No Visible Public Infrastructure Issue" or not ordered:
                issue_detected = False
                confidence = 0.0
                primary_issue = "none"
                logger.warning(f"🚫 AI Guard: No civic issue detected in image. Blocking auto-submission.")
                return primary_issue, "Low", 0.0, "none", "Low", False 

            # Final mappings
            priority = meta.get("final_priority", "Medium")
            severity = "High" if priority == "High" else ("Medium" if priority == "Medium" else "Low")
            
            # Confidence logic from V3
            confidence = 85.0
            known = data.get("known_issues", [])
            unknown = data.get("unknown_issues", [])
            
            for k in known:
                if k.get("issue") == primary_issue:
                    conf_map = {"High": 95.0, "Medium": 75.0, "Low": 45.0}
                    confidence = conf_map.get(k.get("confidence", "High"), 85.0)
                    break
            else:
                for u in unknown:
                    if u.get("issue") == primary_issue:
                        conf_map = {"High": 90.0, "Medium": 65.0, "Low": 35.0} # Lower defaults for unknown
                        confidence = conf_map.get(u.get("confidence", "Medium"), 65.0)
                        break
            
            category = "infrastructure" if "Tier 2" in tier_or_severity else ("public_health" if "Tier 3" in tier_or_severity else "safety")
            
            # 🆕 V4 RECONCILER: Enforce tier-based priority, emergency flags, scenario
            v4_result = reconcile_and_enrich(primary_issue, severity, confidence, category, priority)
            severity = v4_result["severity"]
            priority = v4_result["priority"]
            confidence = v4_result["confidence"]
            
            logger.info(f"✅ Classify V3+V4: {primary_issue} (Tier {v4_result['tier']}, {confidence}%) - {priority} [Scenario {v4_result['scenario']}]")
            if v4_result["emergency_911"]:
                logger.warning(f"🚨 TIER 0 EMERGENCY: {primary_issue} — 911 advisory triggered")
            
            return primary_issue, severity, confidence, category, priority, True

    except Exception as e:
        logger.warning(f"V3 classification failed, using built-in legacy: {e}")

    # Fallback heuristic if Gemini is disabled
    if not GEMINI_API_KEY:
        issue_category_map = await load_json_data("issue_category_map.json")
        description_lower = description.lower()
        issue_keywords = {
            "fire": ["fire", "smoke", "flame", "burn", "blaze"],
            "pothole": ["pothole", "road damage", "crack", "hole", "ft wide", "deep", "swerve"],
            # Do not include "debris" (often appears in fallen-tree scenes)
            "garbage": ["trash", "litter", "garbage", "waste"],
            "property_damage": ["damage", "broken", "destruction"],
            "flood": ["flood", "water", "inundation", "leak"],
            "vandalism": ["graffiti", "vandalism", "deface", "tagging"],
            "structural_damage": ["crack", "collapse", "structural", "foundation"],
            "dead_animal": ["dead animal", "carcass", "roadkill"],
            "tree_fallen": ["fallen tree", "tree uprooted", "tree down", "branch down", "tree on house", "tree on roof", "tree on building"],
            "structural_damage": ["structural", "collapsed", "roof damage", "building damage", "wall damage"],
            "abandoned_vehicle": ["abandoned car", "junk car", "broken down vehicle", "unclaimed vehicle"],
            "animated": ["cartoon", "drawing", "anime", "ai generated", "simulated", "fake", "video game"]
        }
        issue_type = "unknown"
        confidence = 50.0
        for issue, keywords in issue_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                issue_type = issue
                break
        animal_tokens = ["dead animal", "carcass", "roadkill"]
        if any(t in description_lower for t in animal_tokens):
            issue_type = "dead_animal"
        # Description-driven confidence adjustments
        hazard_tokens = [
            "wildfire","house fire","building fire","spreading","spread",
            "out of control","collapse","structural","injury","accident",
            "collision","burst","leak","flood","sirens"
        ]
        controlled_tokens = [
            "campfire","bonfire","bbq","barbecue","fire pit","controlled",
            "festival","diwali","diya","candle","incense","smoke machine","stage"
        ]
        has_hazard = any(t in description_lower for t in hazard_tokens)
        has_controlled = any(t in description_lower for t in controlled_tokens)

        is_animated = any(t in description_lower for t in issue_keywords["animated"])
        
        confidence = 92.0 if issue_type == "pothole" else (85.0 if issue_type != "unknown" else 50.0)
        if is_animated:
            confidence = 10.0
            issue_type = "unknown"
            logger.info(f"Heuristic detected animated/fake content. Confidence set to {confidence}")
        elif has_controlled and not has_hazard:
            confidence = min(confidence, 40.0)
            logger.info(f"Heuristic detected controlled fire. Confidence set to {confidence}")
        elif has_hazard:
            confidence = max(confidence, 88.0)
        # Clamp for non-specific types
        if issue_type in ("unknown", "other"):
            confidence = min(confidence, 50.0)
        high_severity_issues = ["fire", "flood", "flooding", "car_accident", "fire_hazard"]
        high_severity_keywords = ["urgent", "emergency", "critical", "severe", "life threat", "danger"]
        
        medium_severity_issues = ["broken_streetlight", "road_damage", "pothole", "water_leakage", "tree_fallen", "structural_damage"]
        
        low_severity_issues = ["garbage", "trash", "dead_animal", "abandoned_vehicle"]
        
        severity = "Low"
        if issue_type in high_severity_issues or any(k in description_lower for k in high_severity_keywords):
            severity = "High"
        elif issue_type in medium_severity_issues or confidence >= 85:
            severity = "Medium"
        category = issue_category_map.get(issue_type, "public")
        priority = "High" if severity == "High" or confidence > 90 else "Medium"
        logger.info(f"Heuristic classification (no Gemini): {issue_type}, severity {severity}, confidence {confidence}")
        return issue_type, severity, confidence, category, priority, True
    
    last_error = None
    for attempt in range(3):  # V4: 3 attempts with corrective error injection
        try:
            image = Image.open(io.BytesIO(image_content))
            # Use model with key pool rotation
            model, slot = get_gemini_model()

            # Load issue types
            issue_category_map = await load_json_data("issue_category_map.json")
            valid_issue_types = "|".join(issue_category_map.keys()) or "pothole|fire|garbage|flood|vandalism|structural_damage|property_damage"

            prompt = f"""
You are an expert AI trained to classify infrastructure-related issues for a civic reporting app.
Analyze the image and description: "{description}".

STEP 1: REALITY CHECK (CRITICAL)
- Is this a REAL, PHOTO-REALISTIC photo taken by a camera?
- Look for: Cartoon shaders, cel-shading, anime style, perfect smooth textures, AI artifacts, video game HUDs, drawings, digital art, Photoshop edits, or unnatural image manipulations.
- If the image is AI-GENERATED, A VIDEO GAME, A CARTOON, ANIME, SCREENSHOT, MODIFIED, or FAKE:
  - Set "is_real": false
  - Set "confidence": 0
  - Set "issue_type": "unknown"

STEP 2: ISSUE IDENTIFICATION
- Does the image show a VALID Public Infrastructure Issue?
- Issue Types: {valid_issue_types}
- **ROAD DAMAGE ERRORS (CRITICAL)**: 
  - If the image shows potholes, cracks, or broken asphalt, use "pothole" or "road_damage".
  - **DANGER**: Do NOT classify potholes or road work as "car_accident" just because there are orange cones, barriers, or high-viz vests. 
  - ONLY use "car_accident" if you see an actual collison, a damaged car, wreckage, or a car crash. 
  - Traffic cones often mean road maintenance or potholes, not an accident.
- If the image shows a "normal" scene (e.g., a clean road, a happy dog, a normal building, a selfie):
- Set "issue_type": "none"
- Set "confidence": 0 to 60 (Low)
- If the image shows "Fire":
  - controlled fire (bonfire, candle, bbq, festival, diya) -> "issue_type": "controlled_fire", "confidence": 30
  - hazard fire (wildfire, building fire, accident) -> "issue_type": "fire", "confidence": 90+

Return JSON:
{{
  "issue_type": "one of {valid_issue_types}",
  "severity": "high|medium|low",
  "confidence": number (0 to 100),
  "is_real": boolean
}}
"""
            # Run Gemini API call in a separate thread
            timeout = int(os.getenv('AI_TIMEOUT', '25'))
            logger.info(f"Starting Gemini classification via key [{slot.project_name}] (timeout={timeout}s)")

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, [prompt, image, f"Description: {description}"]),
                    timeout=timeout
                )
                get_key_pool().report_success(slot, estimated_tokens=4000)
            except Exception as gen_err:
                get_key_pool().report_error(slot, str(gen_err))
                raise gen_err
            logger.info(f"Gemini classification raw output (attempt {attempt + 1}): {response.text}")
            # Extract and validate JSON
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            json_text = json_match.group(0)
            parsed = json.loads(json_text)
            
            # Parse Reality Check
            is_real = parsed.get("is_real", True)
            if isinstance(is_real, str):
                is_real = is_real.lower() == "true"
            
            issue_type = parsed.get("issue_type", "unknown").lower()
            severity = parsed.get("severity", "Medium").title()
            
            try:
                conf_raw = parsed.get("confidence", 70.0)
                if isinstance(conf_raw, str):
                    # Remove % and other junk
                    conf_raw = re.sub(r'[^0-9.]', '', conf_raw)
                confidence = float(conf_raw)
            except (ValueError, TypeError):
                confidence = 70.0

            if not is_real:
                logger.warning("AI detected FAKE/GENERATED image. Forcing explicit rejection.")
                raise ValueError("FAKE_IMAGE_DETECTED")

            # 2. "None" / "Normal" Filter
            if issue_type in ["none", "normal", "safe", "other", "unknown"]:
                logger.info(f"AI detected no specific issue ('{issue_type}'). Clamping confidence.")
                confidence = min(confidence, 65.0) # Ensure it stays < 70
                issue_type = "unknown"

            # 3. Controlled Fire Filter
            if issue_type == "controlled_fire":
                 logger.info("AI detected controlled fire. Marking as low confidence 'fire'.")
                 issue_type = "fire"
                 confidence = min(confidence, 40.0) # Low confidence for admin review

            # Validate issue_type
            if issue_type not in issue_category_map:
                if issue_type != "unknown":
                    logger.warning(f"Invalid issue_type '{issue_type}' received, defaulting to 'unknown'")
                    issue_type = "unknown"
                    confidence = min(confidence, 70.0)

            # Cross-validate with description (prioritize fallen tree first)
            description_lower = (description + " " + response.text).lower()
            issue_keywords = {
                "fire": ["fire", "smoke", "flame", "burn", "blaze"],
                "pothole": ["pothole", "road damage", "crack in road", "asphalt damage", "broken road", "deep hole"],
                # Do not include generic 'debris' (overlaps with fallen tree scenes)
                "garbage": ["trash", "litter", "garbage", "waste"],
                "property_damage": ["damage", "broken", "destruction"],
                "flood": ["flood", "water", "inundation", "leak"],
                "vandalism": ["graffiti", "vandalism", "deface", "tagging"],
                "structural_damage": ["crack", "collapse", "structural", "foundation"],
                "tree_fallen": [
                    "fallen tree","tree fallen","downed tree","tree down","uprooted",
                    "tree across","tree blocking road","tree blocking the road","tree on road",
                    "branches","branch","trunk","logs","log","limb","limbs",
                    "fallen branches","tree debris","debris from tree"
                ]
            }
            
            # Heuristics: Only apply if valid REAL image and not explicitly marked 'none' by AI
            # (If AI said "none", we trust it more, unless description is VERY specific)
            
            if is_real and confidence > 10:
                # 🛠️ SPECIFIC FIX: If summary or description mentions Pothole, override Car Accident
                if any(tok in description_lower for tok in issue_keywords["pothole"]):
                    if issue_type in ["car_accident", "accident", "unknown", "other"]:
                        logger.info(f"Heuristic detected POTHOLE markers. Overriding {issue_type} to pothole.")
                        issue_type = "pothole"
                        confidence = max(confidence, 85.0)

                # High-priority fallen tree override
                if any(tok in description_lower for tok in issue_keywords["tree_fallen"]):
                    issue_type = "tree_fallen"
                    confidence = max(confidence, 88.0)
                    logger.info(f"Description suggests tree_fallen. Overriding to tree_fallen with confidence {confidence}.")
                else:
                    for issue, keywords in issue_keywords.items():
                        if any(keyword in description_lower for keyword in keywords):
                            if issue_type == "unknown": # Only assist unknown/ambiguous
                                issue_type = issue
                                confidence = max(confidence, 80.0)
                                logger.info(f"Description suggests {issue}. Overriding to {issue} with confidence {confidence}.")
                            break
                            
            animal_tokens = ["dead animal", "carcass", "roadkill"]
            if any(t in description_lower for t in animal_tokens):
                issue_type = "dead_animal"
                if confidence > 10:
                    confidence = max(confidence, 80.0)

            # Description-driven confidence refinements
            hazard_tokens = [
                "wildfire","house fire","building fire","spreading","spread",
                "out of control","collapse","structural","injury","accident",
                "collision","burst","leak","flood","sirens"
            ]
            controlled_tokens = [
                "campfire","bonfire","bbq","barbecue","fire pit","controlled",
                "festival","diwali","diya","candle","incense","smoke machine","stage", "stove"
            ]
            has_hazard = any(t in description_lower for t in hazard_tokens)
            has_controlled = any(t in description_lower for t in controlled_tokens)
            
            if has_controlled and not has_hazard:
                logger.info("Description indicates controlled fire/event. Lowering confidence.")
                confidence = min(confidence, 40.0)
            elif has_hazard and confidence > 10:
                confidence = max(confidence, 88.0)

            # Clamp for non-specific types (Final Guardrail)
            if issue_type in ("unknown", "other", "none"):
                confidence = min(confidence, 65.0)
            
            # Explicitly keep confidence low if it came back very low (likely fake)
            if confidence <= 10:
                logger.info(f"Confidence is low ({confidence}), skipping heuristic overrides.")
            else:
                # Cap confidence
                confidence = min(confidence, 100.0)

            # Determine severity
            high_severity_issues = [
                "fire", "flood", "flooding", "structural_damage", "car_accident", 
                "animal_accident", "sewage_overflow", "unsafe_building", 
                "signal_malfunction", "traffic_signal_issue", "pipe_leak", "no_water_supply"
            ]
            high_severity_keywords = ["urgent", "emergency", "critical", "severe", "life threat", "danger"]
            medium_severity_issues = [
                "pothole", "road_damage", "tree_fallen", "sidewalk_damage", 
                "street_sign_damage", "water_leakage", "abandoned_vehicle", 
                "illegal_construction", "broken_streetlight", "property_damage", "dead_animal"
            ]
            
            # Use AI suggested severity if valid, otherwise fallback to logic
            ai_severity = parsed.get("severity", "").title()
            if ai_severity in ["High", "Medium", "Low"]:
                severity = ai_severity
            else:
                severity = "Low"
                if (issue_type in high_severity_issues or any(k in description_lower for k in high_severity_keywords)) and confidence > 50:
                    severity = "High"
                elif (issue_type in medium_severity_issues or confidence >= 85) and confidence > 50:
                    severity = "Medium"
            
            # Special case for fire: if low confidence/controlled -> Low
            if issue_type == "fire" and confidence < 50:
                 severity = "Low"

            category = issue_category_map.get(issue_type, "public")
            priority = "High" if severity == "High" or confidence > 90 else "Medium"
            
            if confidence < 70:
                priority = "Low"

            logger.info(f"Issue classified as {issue_type} with severity {severity} (confidence: {confidence}, category: {category}, priority: {priority})")
            return issue_type, severity, confidence, category, priority, True
        except ValueError as ve:
            if str(ve) == "FAKE_IMAGE_DETECTED":
                raise ve
            last_error = str(ve)
            logger.warning(f"Attempt {attempt + 1}/3 failed to classify issue: {last_error}")
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))  # V4: 250ms, 500ms backoff
                continue
            return "unknown", "Medium", 50.0, "public", "Medium", True
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1}/3 failed to classify issue: {last_error}")
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))  # V4: backoff before retry
                continue
            return "unknown", "Medium", 50.0, "public", "Medium", True

async def generate_report(
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
    """Generate a detailed report for an infrastructure issue."""
    last_error = None
    for attempt in range(3):  # V4: 3 attempts with corrective error injection
        try:
            # Use model with key pool rotation
            model, slot = get_gemini_model()
            location_str = (
                f"{address}, {zip_code}" if address and address.lower() != "not specified" and zip_code
                else address if address and address.lower() != "not specified"
                else f"Coordinates: {latitude}, {longitude}" if latitude and longitude
                else "Unknown Location"
            )

            # Load department mapping
            issue_department_map = await load_json_data("issue_department_map.json")

            # 🆕 Normalize issue_type to handle variations (lower case and replace spaces with underscores)
            normalized_issue_type = issue_type.lower().replace(" ", "_")

            # 🆕 Try multiple formats for matching
            department = None
            if normalized_issue_type in issue_department_map:
                department = issue_department_map[normalized_issue_type][0]
            elif issue_type.lower() in issue_department_map:
                department = issue_department_map[issue_type.lower()][0]
            elif issue_type in issue_department_map:
                department = issue_department_map[issue_type][0]
            else:
                department = "general"

            # 🆕 Log for debugging
            logger.info(f"Resolved department for issue type '{issue_type}' (normalized: '{normalized_issue_type}'): {department}")
            map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"

            # 🟢 Ticket 3: Authority Logic Adjustment
            # Only fetch authorities if we have a valid, actionable issue
            should_contact_authorities = (
                issue_type.lower() not in ["none", "unknown", "normal", "safe"] 
                and confidence > 45.0
            )

            if should_contact_authorities:
                authority_data = (
                    await asyncio.to_thread(get_authority_by_zip_code, zip_code, issue_type, category) if zip_code
                    else await asyncio.to_thread(get_authority, address, issue_type, latitude, longitude, category)
                )
                responsible_authorities = authority_data.get("responsible_authorities", [{"name": department, "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])
                available_authorities = authority_data.get("available_authorities", [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])

                if available_authorities and "message" in available_authorities[0]:
                    available_authorities = [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}]
            else:
                logger.info(f"Skipping authority lookup for non-issue/low-confidence: {issue_type} ({confidence}%)")
                # 🟢 Ticket 3: If no issue, don't recommend contacting anyone yet
                responsible_authorities = []
                available_authorities = []

            # Get timezone
            timezone_str = await asyncio.to_thread(get_timezone_name, latitude, longitude) or "UTC"
            timezone = pytz.timezone(timezone_str)
            now = datetime.now(timezone)
            local_time = now.strftime("%m/%d/%Y %H:%M")
            utc_time = now.astimezone(pytz.UTC).strftime("%H:%M")
            report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
            report_id = f"eaiser-{now.year}-{report_number}"
            image_filename = f"IMG1_{now.strftime('%Y%m%d_%H%M')}.jpg"

            # Prepare prompt
            decline_prompt = f"- Decline Reason: {decline_reason}\n" if decline_reason else ""
            zip_code_prompt = f"- Zip Code: {zip_code}\n" if zip_code else ""
            prompt = f"""
You are an AI assistant for eaiser AI, generating infrastructure issue reports.
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
- Fire: ["Dispatch fire department immediately.", "Evacuate area if necessary.", "Investigate cause after extinguishing."]
- Garbage: ["Remove debris within 24 hours.", "Install additional trash bins in the area."]
- Flooding: ["Deploy pumps and sandbags immediately.", "Clear drainage systems within 12 hours."]
- Vandalism: ["Remove graffiti within 72 hours.", "Repair damaged property.", "Increase security patrols."]
- Structural damage: ["Cordon off the area immediately.", "Conduct structural inspection within 24 hours.", "Repair or reinforce structure as needed."]
- Property damage: ["Assess extent of damage.", "Contact property owner within 24 hours.", "Arrange repairs within 48 hours."]
Tailor the actions to the specific issue type and context.

If a decline reason is provided, incorporate it into the summary_explanation and add a feedback field in detailed_analysis.
Include the zip code in the summary_explanation if provided.

Return this structure:
{{
  "issue_overview": {{
    "issue_type": "{issue_type.title()}",
    "severity": "{severity.lower()}",
    "confidence": {confidence},
    "category": "{category}",
    "summary_explanation": "A polite, user-friendly summary (max 3 sentences). STATE CLEARLY what visual evidence supports the finding (e.g. 'We detected deep cracking in the asphalt indicating a pothole'). IF NO ISSUE IS FOUND, explain why (e.g. 'The road appears clear and free of obstructions').",
    "admin_analysis": "Technical details for the admin. Mention specific visual features (texture, depth, lighting) that confirmed or rejected the issue.",
    "user_feedback": "A very short, friendly message for the user. (e.g. 'Thank you for reporting this pothole. We have flagged it for review.')"
  }},
  "detailed_analysis": {{
    "root_causes": "Possible causes of the issue.",
    "potential_consequences_if_ignored": "Risks if the issue is not addressed.",
    "public_safety_risk": "low|medium|high",
    "environmental_impact": "none" if issue_type == "pothole" else "low",
    "structural_implications": "low" if issue_type not in ["structural_damage", "property_damage"] else "medium",
    "legal_or_regulatory_considerations": "Road safety regulations." if issue_type == "pothole" else "Local regulations may apply.",
    "feedback": f"User-provided decline reason: {decline_reason}" if decline_reason else null
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
            "confidence": {confidence}
          }}
        }}
Keep the report under 200 words, professional, and specific to the issue type and description.
"""
            # Run Gemini API call
            timeout = int(os.getenv('AI_TIMEOUT', '25'))
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, [prompt, Image.open(io.BytesIO(image_content))]),
                    timeout=timeout
                )
                get_key_pool().report_success(slot, estimated_tokens=8000)
            except Exception as gen_err:
                get_key_pool().report_error(slot, str(gen_err))
                raise gen_err
            logger.info(f"Gemini report output (attempt {attempt + 1}): {response.text}")

            # Extract and validate JSON
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            json_text = json_match.group(0)
            report = json.loads(json_text)

            # Validate report structure
            required_fields = ["issue_overview", "detailed_analysis", "recommended_actions", "responsible_authorities_or_parties", "available_authorities", "additional_notes", "template_fields"]
            missing_fields = [field for field in required_fields if field not in report]
            if missing_fields:
                raise ValueError(f"Missing required fields in report: {missing_fields}")

            # Update report fields
            report["additional_notes"] = f"Location: {location_str}. View live location: {map_link}. Issue ID: {issue_id}. Track report: https://momentum-ai.org/track/{report_id}. Zip Code: {zip_code if zip_code else 'N/A'}."
            report["template_fields"]["map_link"] = map_link
            report["template_fields"]["zip_code"] = zip_code if zip_code else "N/A"
            report["responsible_authorities_or_parties"] = responsible_authorities
            report["available_authorities"] = available_authorities
            
            # UI-friendly aliases for frontend consumption
            issue_overview = report.get("issue_overview", {})
            if "type" not in issue_overview:
                issue_overview["type"] = issue_overview.get("issue_type", issue_type.title())
            if "summary" not in issue_overview:
                issue_overview["summary"] = issue_overview.get(
                    "summary_explanation",
                    f"Issue reported at {location_str}."
                )
            report["issue_overview"] = issue_overview
            
            detailed_analysis = report.get("detailed_analysis", {})
            if "potential_impact" not in detailed_analysis:
                detailed_analysis["potential_impact"] = detailed_analysis.get(
                    "potential_consequences_if_ignored",
                    "Potential risks if ignored."
                )
            report["detailed_analysis"] = detailed_analysis

            issue_overview = report.get("issue_overview", {})
            orig_desc = str(issue_overview.get("summary_explanation") or description or issue_type).strip()

            # Override type to tree_fallen if strong cues found (avoids mislabel as garbage)
            try:
                combined_text = (
                    orig_desc + "\n" +
                    (report.get("detailed_analysis", {}).get("root_causes") or "")
                ).lower()
                tree_tokens = [
                    "fallen tree","tree fallen","downed tree","tree down","uprooted",
                    "tree across","tree blocking road","tree blocking the road","tree on road",
                    "branches","branch","trunk","logs","log","limb","limbs",
                    "branches across","debris from tree","fallen branches","tree debris"
                ]
                if any(tok in combined_text for tok in tree_tokens):
                    issue_overview["type"] = "tree_fallen"
                    try:
                        c = float(issue_overview.get("confidence", 70))
                        issue_overview["confidence"] = max(c, 85)
                    except Exception:
                        issue_overview["confidence"] = 85
            except Exception:
                pass
                
            # Additional safety: Explicit penalty for non-civic / medical / stock photos that leaked
            _im_analysis_check = str(report.get("ai_evaluation", {}).get("image_analysis", "")).lower()
            combo_desc = (str(issue_overview.get("summary_explanation", "")) + _im_analysis_check).lower()
            medical_fake_words = ["medical personnel", "stretcher", "ambulance", "stock photo", "istock", "watermark", "istockphoto"]
            if any(w in combo_desc for w in medical_fake_words):
                 if "collision" not in combo_desc and "crash" not in combo_desc and "infrastructure" not in combo_desc:
                     issue_overview["confidence"] = 15
                     issue_overview["type"] = "None"
                     if "ai_evaluation" in report:
                         report["ai_evaluation"]["issue_detected"] = False
            # Enforce exact summary format to avoid overfitting AI descriptions
            # Pick a short 1-line description of the specific issue
            orig_desc = str(issue_overview.get("summary_explanation") or description or issue_type).strip()
            ai_eval = report.get("ai_evaluation", {})
            # Prevent generic frontend strings from overriding the AI analysis
            if "user reported issue" in orig_desc.lower():
                _im_analysis = ai_eval.get("image_analysis", "")
                if _im_analysis and _im_analysis != "No explicit image analysis provided.":
                    orig_desc = str(_im_analysis).strip()
                else:
                    orig_desc = str(issue_overview.get('type', issue_type)).title()
                
            entry_issue_desc = orig_desc.split(".")[0].strip()
            if len(entry_issue_desc) > 120:
                entry_issue_desc = entry_issue_desc[:117] + "..."
            if not entry_issue_desc:
                entry_issue_desc = issue_overview.get('type', issue_type).title()
            
            lower_desc = entry_issue_desc.lower()
            if lower_desc.startswith("possible "): entry_issue_desc = entry_issue_desc[9:]
            elif lower_desc.startswith("identified "): entry_issue_desc = entry_issue_desc[11:]
            elif lower_desc.startswith("a potential "): entry_issue_desc = entry_issue_desc[12:]
            elif lower_desc.startswith("potential "): entry_issue_desc = entry_issue_desc[10:]
            elif lower_desc.startswith("a "): entry_issue_desc = entry_issue_desc[2:]
            
            exact_summary = f"Possible {entry_issue_desc} has been reported at {location_str}; priority {priority}, confidence {confidence}."
            issue_overview["summary_explanation"] = exact_summary
            issue_overview["summary"] = exact_summary

            # Ensure alias fields reflect final summary (Format enforced)
            report["issue_overview"] = issue_overview

            # 🆕 V4 TIER 0 SAFETY NET: Emergency issues MUST have issue_detected=true
            # This prevents the "No Issue Detected" screen for fire/accident/power line
            tier0_labels = {"fire_hazard", "car_accident", "downed_power_line", "fire"}
            detected_type = str(issue_overview.get("type", "") or issue_overview.get("issue_type", "")).lower().replace(" ", "_")
            if detected_type in tier0_labels or issue_type.lower().replace(" ", "_") in tier0_labels:
                if "ai_evaluation" not in report:
                    report["ai_evaluation"] = {}
                report["ai_evaluation"]["issue_detected"] = True
                report["ai_evaluation"]["detected_issue_type"] = detected_type
                # Ensure confidence is high for clear emergencies
                try:
                    c = float(issue_overview.get("confidence", 85))
                    if c < 85:
                        issue_overview["confidence"] = 85
                        report["issue_overview"] = issue_overview
                except Exception:
                    pass
                logger.info(f"🚨 V4 Tier 0 safety net: Forced issue_detected=true for {detected_type}")

            logger.info(f"Report generated for issue {issue_id} with issue_type {issue_type}")
            return report
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1}/3 failed to generate report: {last_error}")
            if attempt < 2:
                await asyncio.sleep(0.25 * (attempt + 1))  # V4: 250ms, 500ms backoff
                continue
            break  # Exhausted all retries, fall through to fallback

    # Fallback report
    timezone_str = await asyncio.to_thread(get_timezone_name, latitude, longitude) or "UTC"
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)
    local_time = now.strftime("%Y-%m-%d %H:%M")
    utc_time = now.astimezone(pytz.UTC).strftime("%H:%M")
    report_number = str(int(now.strftime("%Y%m%d%H%M%S")) % 1000000).zfill(6)
    report_id = f"eaiser-{now.year}-{report_number}"
    image_filename = f"IMG1_{now.strftime('%Y%m%d_%H%M')}.jpg"
    map_link = f"https://www.google.com/maps?q={latitude},{longitude}" if latitude and longitude else "Coordinates unavailable"
    location_str = (
        f"{address}, {zip_code}" if address and address.lower() != "not specified" and zip_code
        else address if address and address.lower() != "not specified"
        else f"Coordinates: {latitude}, {longitude}" if latitude and longitude
        else "Unknown Location"
    )

    authority_data = (
        await asyncio.to_thread(get_authority_by_zip_code, zip_code, issue_type, category) if zip_code
        else await asyncio.to_thread(get_authority, address, issue_type, latitude, longitude, category)
    )
    responsible_authorities = authority_data.get("responsible_authorities", [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])
    available_authorities = authority_data.get("available_authorities", [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}])

    if available_authorities and "message" in available_authorities[0]:
        available_authorities = [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}]

    # Issue-specific recommended actions
    if issue_type == "fire":
        actions = ["Dispatch fire department immediately.", "Evacuate area if necessary.", "Investigate cause after extinguishing."]
    elif issue_type == "pothole":
        actions = ["Fill pothole and mark with cones within 48 hours.", "Conduct follow-up inspection after repair."]
    elif issue_type == "garbage":
        actions = ["Remove debris within 24 hours.", "Install additional trash bins in the area."]
    elif issue_type == "flood":
        actions = ["Deploy pumps and sandbags immediately.", "Clear drainage systems within 12 hours."]
    elif issue_type == "vandalism":
        actions = ["Remove graffiti within 72 hours.", "Repair damaged property.", "Increase security patrols."]
    elif issue_type == "structural_damage":
        actions = ["Cordon off the area immediately.", "Conduct structural inspection within 24 hours.", "Repair or reinforce structure as needed."]
    elif issue_type == "property_damage":
        actions = ["Assess extent of damage.", "Contact property owner within 24 hours.", "Arrange repairs within 48 hours."]
    else:
        actions = [f"Notify the {department} for immediate action.", "Conduct a professional inspection."]

    report = {
        "issue_overview": {
            "issue_type": issue_type.title(),
            "severity": severity.lower(),
            "confidence": confidence,
            "category": category,
            "summary_explanation": f"Possible {issue_type.title()} has been reported at {location_str}; priority {priority}, confidence {confidence:.1f}%."
        },
        "detailed_analysis": {
            "root_causes": "Wear and tear or heavy traffic." if issue_type == "pothole" else "Undetermined; requires inspection.",
            "potential_consequences_if_ignored": "Vehicle damage or accidents." if issue_type == "pothole" else "Potential safety or compliance risks.",
            "public_safety_risk": severity.lower(),
            "environmental_impact": "none" if issue_type == "pothole" else "low",
            "structural_implications": "low" if issue_type not in ["structural_damage", "property_damage"] else "medium",
            "legal_or_regulatory_considerations": "Road safety regulations." if issue_type == "pothole" else "Local regulations may apply.",
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
            "confidence": confidence
        }
    }
    if decline_reason:
        report["issue_overview"]["summary_explanation"] += f"\nDecline reason: {decline_reason}."

    # Enforce exact summary format in fallback response
    issue_overview = report.get("issue_overview", {})
    if "type" not in issue_overview:
        issue_overview["type"] = issue_overview.get("issue_type", issue_type.title())
        
    orig_desc = str(issue_overview.get("summary_explanation") or description or issue_type).strip()
    ai_eval = report.get("ai_evaluation", {})
    if "user reported issue" in orig_desc.lower():
        _im_analysis = ai_eval.get("image_analysis", "")
        if _im_analysis and _im_analysis != "No explicit image analysis provided.":
            orig_desc = str(_im_analysis).strip()
        else:
            orig_desc = str(issue_overview.get('type', issue_type)).title()

    entry_issue_desc = orig_desc.split(".")[0].strip()
    if len(entry_issue_desc) > 120:
        entry_issue_desc = entry_issue_desc[:117] + "..."
    if not entry_issue_desc:
        entry_issue_desc = issue_overview.get('type', issue_type).title()
    lower_desc = entry_issue_desc.lower()
    if lower_desc.startswith("possible "): entry_issue_desc = entry_issue_desc[9:]
    elif lower_desc.startswith("identified "): entry_issue_desc = entry_issue_desc[11:]
    elif lower_desc.startswith("a potential "): entry_issue_desc = entry_issue_desc[12:]
    elif lower_desc.startswith("potential "): entry_issue_desc = entry_issue_desc[10:]
    elif lower_desc.startswith("a "): entry_issue_desc = entry_issue_desc[2:]
        
    exact_summary = f"Possible {entry_issue_desc} has been reported at {location_str}; priority {priority}, confidence {confidence}."
    
    # 🟢 Preserve original detailed summary for frontend/detailed report sections
    issue_overview["detailed_description"] = orig_desc
    
    issue_overview["summary_explanation"] = exact_summary
    issue_overview["summary"] = exact_summary
    report["issue_overview"] = issue_overview

    md = report.get("detailed_analysis", {})
    if "potential_impact" not in md:
        md["potential_impact"] = md.get("potential_consequences_if_ignored", "Potential risks if ignored.")
    report["detailed_analysis"] = md

    logger.info(f"Fallback report generated for issue {issue_id} with issue_type {issue_type}")
    return report


class AIService:
    async def analyze_issue(self, description: str, issue_type: str, severity: str) -> Dict[str, Any]:
        for attempt in range(2):
            try:
                model, slot = get_gemini_model()
                prompt = (
                    "You are an expert civil infrastructure AI. Given an issue description, "
                    "type and severity, produce JSON with: summary (string), risk_level (low|medium|high), "
                    "confidence (0-100), recommended_actions (array of 3 short actions). Only JSON.\n"
                    f"Description: {description}\nIssue Type: {issue_type}\nSeverity: {severity}"
                )
                response = await asyncio.to_thread(model.generate_content, prompt)
                get_key_pool().report_success(slot, estimated_tokens=2000)
                text = response.text or ""
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    return json.loads(match.group(0))
                # Fallback simple analysis if parsing failed
                break
            except Exception:
                await asyncio.sleep(0.5)
        # Heuristic fallback without AI
        description_lower = description.lower()
        risk_level = "high" if any(k in description_lower for k in ["urgent", "emergency", "critical", "severe"]) else (
            "medium" if any(k in description_lower for k in ["risk", "unsafe", "accident"]) else "low"
        )
        actions = [
            "Log and prioritize in maintenance queue",
            "Notify responsible department",
            "Schedule inspection within 24-72 hours"
        ]
        return {
            "summary": f"Issue '{issue_type}' with {severity} severity. Description analyzed.",
            "risk_level": risk_level,
            "confidence": 80,
            "recommended_actions": actions
        }


def get_ai_service() -> AIService:
    return AIService()
