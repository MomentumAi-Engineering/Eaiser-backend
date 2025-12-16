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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY missing in environment variables")
            # We might allow initialization without key but methods will fail
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model_name = "gemini-2.0-flash" 
            try:
                self.model = genai.GenerativeModel(self.model_name)
            except Exception:
                self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def detect_fake(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Returns:
        {
            "is_fake": True/False,
            "confidence": number (0-100),
            "reason": "text explanation"
        }
        """
        if not self.model:
             return {"is_fake": False, "confidence": 0, "reason": "No API Key"}

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
            image = Image.open(io.BytesIO(image_bytes))
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, image]
            )

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
