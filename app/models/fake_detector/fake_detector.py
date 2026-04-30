import google.generativeai as genai
import os
import asyncio
import json
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FakeDetector:
    def __init__(self):
        # 🚀 Uses key pool now — model is fetched per-call, not cached
        self.model = None  # Will be fetched from pool on each detect_fake() call
        self.model_name = "gemini-2.0-flash"
        try:
            from services.gemini_key_pool import get_key_pool
            pool = get_key_pool()
            if pool.key_count > 0:
                logger.info(f"FakeDetector initialized with key pool ({pool.key_count} keys)")
            else:
                logger.warning("FakeDetector: No keys in pool")
        except Exception as e:
            logger.warning(f"FakeDetector init warning: {e}")

    async def detect_fake(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Returns:
        {
            "is_fake": True/False,
            "confidence": number (0-100),
            "reason": "text explanation"
        }
        """
        prompt = """
Analyze this image and determine if it appears to be a Fake, Manipulated, or Irrelevant (spam/cartoon/not infrastructure) image.
We are looking for real photos of civic issues (potholes, garbage, etc.).
Return STRICT JSON only:
{
  "is_fake": true,
  "confidence": 0-100,
  "reason": "Why it is fake or real"
}
"""
        try:
            from PIL import Image
            import io
            from services.gemini_key_pool import get_key_pool
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # 🚀 Get model from key pool (auto-rotates keys)
            pool = get_key_pool()
            model, slot = pool.get_model(model_name=self.model_name)
            
            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, image]
            )
            pool.report_success(slot, estimated_tokens=2000)

            raw = response.text
            # Robust JSON extraction
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                json_text = match.group(0)
                return json.loads(json_text)
            else:
                 return {"is_fake": False, "confidence": 0, "reason": "JSON parse error"}

        except Exception as e:
            logger.warning(f"Fake detector failed: {e}")
            try:
                pool.report_error(slot, str(e))
            except Exception:
                pass
            return {
                "is_fake": False,
                "confidence": 0,
                "reason": f"Error: {str(e)}"
            }

_fake_detector = None

def get_fake_detector() -> FakeDetector:
    global _fake_detector
    if _fake_detector is None:
        _fake_detector = FakeDetector()
    return _fake_detector
