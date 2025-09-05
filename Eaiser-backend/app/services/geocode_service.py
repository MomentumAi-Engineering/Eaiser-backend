import requests
import logging
import re
import os
from dotenv import load_dotenv
from typing import Dict, Any
from fastapi import HTTPException

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY"

async def geocode_zip_code(zip_code: str) -> Dict[str, Any]:
    """
    Geocode a US zip code to retrieve address, latitude, and longitude using Google Maps Geocoding API.

    Args:
        zip_code: A 5-digit US zip code (e.g., "90210").

    Returns:
        Dict with keys: address (str), latitude (float), longitude (float), zip_code (str).

    Raises:
        ValueError: If zip_code is invalid or GOOGLE_API_KEY is not set.
        HTTPException: If geocoding fails in production.
    """
    if not zip_code or not re.match(r"^\d{5}$", zip_code):
        logger.error(f"Invalid zip code format: {zip_code}")
        raise ValueError("Zip code must be a 5-digit US zip code")

    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logger.error("Invalid Google API key. Set GOOGLE_API_KEY in environment variables.")
        raise ValueError("Invalid Google API key")

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": zip_code,
            "key": GOOGLE_API_KEY
        }
        logger.debug(f"Geocoding zip code: {zip_code}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "OK":
            logger.error(f"Zip code geocoding failed: {data['status']}")
            if os.getenv("ENV") == "production":
                raise HTTPException(status_code=500, detail=f"Zip code geocoding failed: {data['status']}")
            return {"address": "Unknown Address", "latitude": 0.0, "longitude": 0.0, "zip_code": zip_code}

        result = data["results"][0]
        location = result["geometry"]["location"]
        formatted_zip = ""
        for component in result["address_components"]:
            if "postal_code" in component["types"]:
                formatted_zip = component["long_name"]
                break

        logger.info(f"Geocoded zip code {zip_code}: {result['formatted_address']}, {location['lat']}, {location['lng']}")
        return {
            "address": result["formatted_address"],
            "latitude": location["lat"],
            "longitude": location["lng"],
            "zip_code": formatted_zip or zip_code
        }
    except Exception as e:
        logger.error(f"Zip code geocoding error for {zip_code}: {str(e)}")
        if os.getenv("ENV") == "production":
            raise HTTPException(status_code=500, detail=f"Zip code geocoding error: {str(e)}")
        return {"address": "Unknown Address", "latitude": 0.0, "longitude": 0.0, "zip_code": zip_code}

def geocode_address(address: str) -> Dict[str, Any]:
    """
    Geocode an address to retrieve latitude, longitude, and zip code using Google Maps Geocoding API.

    Args:
        address: The address to geocode.

    Returns:
        Dict with keys: address (str), latitude (float), longitude (float), zip_code (str).
    """
    if not address:
        logger.warning("No address provided for geocoding")
        return {"latitude": 0.0, "longitude": 0.0, "address": "Unknown Address", "zip_code": ""}

    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logger.error("Invalid Google API key. Set GOOGLE_API_KEY in environment variables.")
        raise ValueError("Invalid Google API key")

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address,
            "key": GOOGLE_API_KEY
        }
        logger.debug(f"Geocoding address: {address}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "OK":
            logger.error(f"Geocoding failed for address {address}: {data['status']}")
            if os.getenv("ENV") == "production":
                raise HTTPException(status_code=500, detail=f"Geocoding failed: {data['status']}")
            return {"latitude": 0.0, "longitude": 0.0, "address": address, "zip_code": ""}

        result = data["results"][0]
        location = result["geometry"]["location"]
        zip_code = ""
        for component in result["address_components"]:
            if "postal_code" in component["types"]:
                zip_code = component["long_name"]
                break

        logger.info(f"Geocoded address {address}: {result['formatted_address']}, {location['lat']}, {location['lng']}, zip: {zip_code}")
        return {
            "latitude": location["lat"],
            "longitude": location["lng"],
            "address": result["formatted_address"],
            "zip_code": zip_code
        }
    except Exception as e:
        logger.error(f"Geocoding error for address {address}: {str(e)}")
        if os.getenv("ENV") == "production":
            raise HTTPException(status_code=500, detail=f"Geocoding error: {str(e)}")
        return {"latitude": 0.0, "longitude": 0.0, "address": address, "zip_code": ""}

def reverse_geocode(latitude: float, longitude: float) -> Dict[str, str]:
    """
    Reverse geocode coordinates to retrieve address and zip code using Google Maps Geocoding API.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.

    Returns:
        Dict with keys: address (str), zip_code (str).
    """
    if not latitude or not longitude:
        logger.warning("Invalid coordinates provided for reverse geocoding")
        return {"address": "Unknown Address", "zip_code": ""}

    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logger.error("Invalid Google API key. Set GOOGLE_API_KEY in environment variables.")
        raise ValueError("Invalid Google API key")

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "latlng": f"{latitude},{longitude}",
            "key": GOOGLE_API_KEY
        }
        logger.debug(f"Reverse geocoding coordinates: {latitude}, {longitude}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "OK":
            logger.error(f"Reverse geocoding failed for {latitude}, {longitude}: {data['status']}")
            if os.getenv("ENV") == "production":
                raise HTTPException(status_code=500, detail=f"Reverse geocoding failed: {data['status']}")
            return {"address": "Unknown Address", "zip_code": ""}

        result = data["results"][0]
        zip_code = ""
        for component in result["address_components"]:
            if "postal_code" in component["types"]:
                zip_code = component["long_name"]
                break

        logger.info(f"Reverse geocoded {latitude}, {longitude}: {result['formatted_address']}, zip: {zip_code}")
        return {
            "address": result["formatted_address"],
            "zip_code": zip_code
        }
    except Exception as e:
        logger.error(f"Reverse geocoding error for {latitude}, {longitude}: {str(e)}")
        if os.getenv("ENV") == "production":
            raise HTTPException(status_code=500, detail=f"Reverse geocoding error: {str(e)}")
        return {"address": "Unknown Address", "zip_code": ""}