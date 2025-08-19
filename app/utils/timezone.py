import logging
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

def get_timezone_name(latitude: float, longitude: float) -> str:
    """Get timezone name based on coordinates using timezonefinder."""
    try:
        tf = TimezoneFinder()
        timezone_name = tf.timezone_at(lat=latitude, lng=longitude)
        if not timezone_name:
            logger.warning(f"No timezone found for coordinates ({latitude}, {longitude}), falling back to UTC")
        return timezone_name or "UTC"
    except Exception as e:
        logger.error(f"Timezone lookup failed for coordinates ({latitude}, {longitude}): {str(e)}", exc_info=True)
        return "UTC"