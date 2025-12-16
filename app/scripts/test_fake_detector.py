import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.fake_detector.fake_detector import FakeDetector

async def test():
    print("Testing FakeDetector...")
    
    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set in environment.")
        return

    detector = FakeDetector()
    
    # Path to a sample image
    # Adjust this path if needed
    img_path = Path(__file__).parent.parent / "dataset" / "real" / "broken_streetlight" / "images" / "1.jpg"
    
    if not img_path.exists():
        print(f"❌ Image not found at {img_path}")
        # Try pothole if broken_streetlight fails
        img_path = Path(__file__).parent.parent / "dataset" / "real" / "pothole" / "images" / "China_Drone_000057_jpg.rf.99130ec888b96560931fd1bda936ff28.jpg"
        if not img_path.exists():
             print(f"❌ Image not found at {img_path}")
             return

    print(f"📸 Testing with image: {img_path.name}")
    
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        
    result = await detector.detect_fake(img_bytes)
    
    print("\n🔍 Result:")
    print(result)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Load env vars if .env exists
    asyncio.run(test())
