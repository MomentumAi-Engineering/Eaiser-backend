import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("keep_alive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# URL to ping (Update this with your Render URL)
# Example: "https://your-backend-service.onrender.com/health"
URL = "http://localhost:8000/health"  # Default to local for testing, change for prod

def ping_server():
    try:
        start_time = time.time()
        response = requests.get(URL, timeout=10)
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            logger.info(f"✅ Success: {response.status_code} | Latency: {elapsed:.2f}ms")
        else:
            logger.warning(f"⚠️ Warning: Received status {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Error pinging server: {str(e)}")

if __name__ == "__main__":
    logger.info(f"🚀 Starting Keep-Alive Pinger for {URL}")
    logger.info("Ping interval: 5 minutes")
    
    while True:
        ping_server()
        # Sleep for 5 minutes (300 seconds)
        # This prevents Render/Heroku from spinning down inactive free/starter tiers
        time.sleep(300)
