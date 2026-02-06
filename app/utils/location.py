from typing import List, Dict
import logging
import json
from pathlib import Path
from utils.timezone import get_timezone_name

logger = logging.getLogger(__name__)

def load_json_data(file_name: str) -> dict:
    """Load JSON data from a file."""
    try:
        file_path = Path(__file__).parent.parent / "data" / file_name
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {str(e)}")
        return {}

def _canonical_issue(issue_type: str) -> str:
    """Normalize and canonicalize issue_type to match data map keys."""
    s = (issue_type or "unknown").strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    
    # Heuristic substring matching for robust classification
    if any(k in s for k in ["pothole", "crack", "hole", "cracked"]): return "road_damage"
    if any(k in s for k in ["fire", "smoke", "blaze", "burn"]): return "fire"
    if any(k in s for k in ["flood", "waterlog", "water_logging"]): return "flood"
    if any(k in s for k in ["animal", "carcass", "dead", "roadkill"]): return "dead_animal"
    if any(k in s for k in ["garbage", "trash", "waste", "litter"]): return "garbage"
    if any(k in s for k in ["streetlight", "street_light", "lamp"]): return "broken_streetlight"
    if any(k in s for k in ["vandalism", "graffiti", "paint"]): return "vandalism"
    if any(k in s for k in ["tree", "fallen", "branch"]): return "tree_fallen"
    if any(k in s for k in ["leak", "burst", "pipe"]): return "water_leakage"
    
    alias = {
        "road_hole": "road_damage",
        "damaged_road": "road_damage",
        "road_broken": "road_damage",
        "building_fire": "fire",
        "street_flood": "flood",
        "uncovered_drain": "open_drain",
        "clogged_drain": "blocked_drain",
        "animal_on_road": "roadkill",
        "fallen_tree": "tree_fallen",
    }
    return alias.get(s, s)

def get_authority(address: str, issue_type: str, latitude: float, longitude: float, category: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Determine the relevant authorities for an issue based on issue type, category, and coordinates.
    Returns a dictionary with responsible and available authorities, each as a list of dictionaries with department details (name, email, type, timezone).
    """
    try:
        issue_type = _canonical_issue(issue_type)
        raw_cat = load_json_data("issue_category_map.json")
        issue_category_map = {str(k).lower(): v for k, v in raw_cat.items()}  # normalize keys
        issue_category = category if category else issue_category_map.get(issue_type, "public")
        timezone = get_timezone_name(latitude, longitude) or "UTC"
        logger.debug(f"Resolved timezone for coordinates ({latitude}, {longitude}): {timezone}")

        # Default authorities for fallback
        default_authorities = [
            {"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": timezone}
        ]
        default_available = default_authorities + [
            {"name": "General Support", "email": "chrishabh2002@gmail.com", "type": "business_support", "timezone": timezone}
        ]

        # Map issue categories to responsible and available authorities
        authority_map = {
            "public": {
                "responsible_authorities": default_authorities,
                "available_authorities": default_available
            },
            "business": {
                "responsible_authorities": [{"name": "Business Support", "email": "chrishabh2002@gmail.com", "type": "business_support", "timezone": timezone}],
                "available_authorities": default_available
            },
            "public_and_business": {
                "responsible_authorities": default_authorities,
                "available_authorities": default_available
            }
        }

        result = authority_map.get(issue_category, {
            "responsible_authorities": default_authorities,
            "available_authorities": default_available
        })

        # Enhance fallback using department mapping when zip is not provided
        try:
            dept_map = load_json_data("issue_department_map.json")
            departments = dept_map.get(issue_type, dept_map.get(_canonical_issue(issue_type), [])) or []
            if isinstance(departments, str):
                departments = [departments]
            dept_defaults = {
                "public_works": "City Public Works",
                "sanitation": "Sanitation Department",
                "fire": "Fire Department",
                "police": "Police Department",
                "animal_control": "Animal Control",
                "environment": "Environmental Department",
                "water_utility": "Water Utility",
                "code_enforcement": "Code Enforcement",
                "transportation": "Transportation Department",
                "building_inspection": "Building Inspection",
                "property_management": "Property Management",
                "emergency": "Emergency Services",
                "general": "City Department"
            }
            # Build department-based authorities (generic emails/timezone)
            dept_authorities = [
                {"name": dept_defaults.get(d, d.replace("_", " ").title()), "email": "eaiser@momntumai.com", "type": d, "timezone": timezone}
                for d in departments if isinstance(d, str)
            ]
            if dept_authorities:
                # Prepend first department as responsible lead
                result["responsible_authorities"] = dept_authorities[:1] + result["responsible_authorities"]
                # Extend available authorities with all departments
                result["available_authorities"] = dept_authorities + result["available_authorities"]
        except Exception:
            pass
        logger.info(f"Authorities for issue type {issue_type} at {address} (timezone: {timezone}): {[auth['name'] for auth in result['responsible_authorities']]}")
        return result
    except Exception as e:
        logger.error(f"Failed to determine authorities for issue type {issue_type}: {str(e)}")
        return {
            "responsible_authorities": [],
            "available_authorities": [{"name": "City Department", "email": "chrishabh1000@gmail.com", "type": "general", "timezone": "UTC"}]
        }

def get_authority_by_zip_code(zip_code: str, issue_type: str, category: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Determine authorities based on zip code, issue type, and category.
    Returns a dictionary with responsible and available authorities.
    Strictly adheres to issue_department_map.json for responsible authorities.
    """
    try:
        # Clean and validate zip code format (5 digits)
        if zip_code:
            zip_code = str(zip_code).strip().split('-')[0][:5]
            
        if not zip_code or not zip_code.isdigit() or len(zip_code) != 5:
            logger.warning(f"Invalid or missing zip code: {zip_code}. Falling back to general.")
            return {
                "responsible_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}],
                "available_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
            }

        # Normalize issue type
        issue_type = _canonical_issue(issue_type)
        
        # Load mappings
        dept_map = load_json_data("issue_department_map.json")
        # Ensure keys are normalized (though usually they should be in the file)
        issue_department_map = {str(k).lower(): v for k, v in dept_map.items()}
        
        # Strict lookup for departments
        # If issue type is not found, default to 'unknown' mapping or 'general'
        departments = issue_department_map.get(issue_type, issue_department_map.get("unknown", ["general"]))

        zip_code_authorities = load_json_data("zip_code_authorities.json")
        zip_key = zip_code if zip_code in zip_code_authorities else None
        
        if not zip_key:
            logger.warning(f"Zip code {zip_code} not found in database. Falling back to general.")
            return {
                "responsible_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}],
                "available_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
            }

        zip_data = zip_code_authorities[zip_key]

        # 1. Strictly populate responsible authorities based on the map
        responsible_authorities = []
        for dept in departments:
            if dept in zip_data:
                responsible_authorities.extend(zip_data[dept])
        
        # 2. Populate available authorities with ALL departments in that zip code
        available_authorities = []
        for dept_list in zip_data.values():
            available_authorities.extend(dept_list)

        # 3. Add General Support/Business Support as an option in available authorities
        # Use the timezone from the first responsible authority, or default to UTC
        ref_timezone = responsible_authorities[0]["timezone"] if responsible_authorities else "UTC"
        
        available_authorities.append({
            "name": "General Support",
            "email": "chrishabh2002@gmail.com",
            "type": "business_support",
            "timezone": ref_timezone
        })

        # Remove duplicates while preserving order
        def unique_auths(auth_list):
            seen = set()
            return [auth for auth in auth_list if not (auth["email"] in seen or seen.add(auth["email"]))]

        unique_responsible = unique_auths(responsible_authorities)
        unique_available = unique_auths(available_authorities)

        if not unique_responsible:
            logger.warning(f"No matching authorities for zip code {zip_code} and issue type {issue_type}. Falling back.")
            return {
                "responsible_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}],
                "available_authorities": [{"name": "City Department", "email": "eaiser@momntumai.com", "type": "general"}]
            }

        result = {
            "responsible_authorities": unique_responsible,
            "available_authorities": unique_available
        }
        logger.info(f"Authorities for zip code {zip_code} and issue type {issue_type}: {[auth['name'] for auth in unique_responsible]}")
        return result
    except Exception as e:
        logger.error(f"Failed to determine authorities for zip code {zip_code} and issue type {issue_type}: {str(e)}")
        return {
            "responsible_authorities": [],
            "available_authorities": [{"message": "eaiser services are not available in this area, coming soon in the future"}]
        }
