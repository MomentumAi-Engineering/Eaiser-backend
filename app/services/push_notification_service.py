import httpx
import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

async def send_expo_push_notification(push_token: str, title: str, body: str, data: Optional[Dict[str, Any]] = None):
    """
    Send a push notification via Expo Push Service.
    """
    if not push_token:
        return False
        
    if not push_token.startswith("ExponentPushToken"):
        # Could be a different platform or invalid token
        logger.warning(f"Push token does not look like an Expo token: {push_token}")
        # We try anyway if it's non-empty, maybe it's raw FCM/APNS (though Expo suggests ExponentPushToken)
        # return False

    payload = {
        "to": push_token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data or {},
        "priority": "high",
        "channelId": "default",
        "color": "#C8A84E"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post("https://exp.host/--/api/v2/push/send", json=payload)
            if response.status_code == 200:
                res_data = response.json()
                # Expo returns 200 even if individual tokens fail
                # The response contains a list of 'data' objects with 'status'
                logger.info(f"✅ Push notification request accepted by Expo for token {push_token[:15]}...")
                return True
            else:
                logger.error(f"❌ Expo Push API failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Exception sending Expo Push: {e}")
            return False

def trigger_push_notification(push_token: Optional[str], title: str, body: str, data: Optional[Dict[str, Any]] = None):
    """
    Fire-and-forget push notification trigger.
    """
    if not push_token:
        return
        
    # Run in background task using asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(send_expo_push_notification(push_token, title, body, data))
        else:
            # For scripts or non-loop environments
            asyncio.run(send_expo_push_notification(push_token, title, body, data))
    except Exception as e:
        logger.error(f"Failed to schedule push notification task: {e}")
