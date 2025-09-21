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

def get_authority(address: str, issue_type: str, latitude: float, longitude: float, category: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Determine the relevant authorities for an issue based on issue type, category, and coordinates.
    Returns a dictionary with responsible and available authorities, each as a list of dictionaries with department details (name, email, type, timezone).
    """
    try:
        issue_type = issue_type.lower() if issue_type else "unknown"
        issue_category_map = load_json_data("issue_category_map.json")
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
    Returns a dictionary with responsible and available authorities, each as a list of dictionaries with department details (name, email, type, timezone).
    """
    try:
        # Validate zip code format (5 digits)
        if not zip_code or not zip_code.isdigit() or len(zip_code) != 5:
            logger.warning(f"Invalid zip code format: {zip_code}. Returning unavailable message.")
            return {
                "responsible_authorities": [],
                "available_authorities": [{"message": "Snapfix services are not available in this area, coming soon in the future"}]
            }

        issue_type = issue_type.lower() if issue_type else "unknown"
        issue_category_map = load_json_data("issue_category_map.json")
        issue_category = category if category else issue_category_map.get(issue_type, "public")
        issue_department_map = load_json_data("issue_department_map.json")
        departments = issue_department_map.get(issue_type, ["general"])

        zip_code_authorities = load_json_data("zip_code_authorities.json")
        zip_key = zip_code if zip_code in zip_code_authorities else None
        if not zip_key:
            logger.warning(f"Zip code {zip_code} not found in database. Returning unavailable message.")
            return {
                "responsible_authorities": [],
                "available_authorities": [{"message": "Snapfix services are not available in this area, coming soon in the future"}]
            }

        # Get responsible authorities from zip_code_authorities (issue-specific)
        responsible_authorities = []
        for dept in departments:
            if dept in zip_code_authorities.get(zip_key, {}):
                dept_authorities = zip_code_authorities[zip_key][dept]
                responsible_authorities.extend(dept_authorities)
        
        # Get ALL available authorities from zip_code_authorities (all departments for that zip code)
        available_authorities = []
        for dept_name, dept_authorities in zip_code_authorities.get(zip_key, {}).items():
            available_authorities.extend(dept_authorities)

        # Add additional authorities for selection (e.g., general or business support)
        available_authorities.extend([
            {"name": "General Support", "email": "chrishabh2002@gmail.com", "type": "business_support", "timezone": responsible_authorities[0]["timezone"] if responsible_authorities else "UTC"}
        ])

        # Remove duplicates while preserving order
        seen = set()
        unique_responsible = [auth for auth in responsible_authorities if not (auth["email"] in seen or seen.add(auth["email"]))]
        unique_available = [auth for auth in available_authorities if not (auth["email"] in seen or seen.add(auth["email"]))]

        if not unique_responsible:
            logger.warning(f"No matching authorities for zip code {zip_code} and issue type {issue_type}. Returning unavailable message.")
            return {
                "responsible_authorities": [],
                "available_authorities": [{"message": "Snapfix services are not available in this area, coming soon in the future"}]
            }

        result = {
            "responsible_authorities": unique_responsible,
            "available_authorities": unique_available
        }
        logger.info(f"Authorities for zip code {zip_code} and issue type {issue_type} (timezone: {unique_responsible[0]['timezone'] if unique_responsible else 'UTC'}): {[auth['name'] for auth in unique_responsible]}")
        return result
    except Exception as e:
        logger.error(f"Failed to determine authorities for zip code {zip_code} and issue type {issue_type}: {str(e)}")
        return {
            "responsible_authorities": [],
            "available_authorities": [{"message": "Snapfix services are not available in this area, coming soon in the future"}]
        }