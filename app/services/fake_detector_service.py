from typing import Dict, Any
from io import BytesIO
from PIL import Image, ImageStat

def detect_fake_or_flag(image_bytes: bytes) -> Dict[str, Any]:
    """
    Lightweight heuristic to classify image as real/fake/flag.
    - fake: extremely low variance + tiny resolution or extreme aspect ratio
    - flag: suspicious patterns but not conclusively fake
    - real: otherwise
    Returns: { classification: 'real'|'fake'|'flag', confidence: int, reason: str }
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        pixels = max(1, w * h)
        aspect = max(w, h) / max(1, min(w, h))
        stat_l = ImageStat.Stat(img.convert("L"))
        var = sum(stat_l.var) / max(1, len(stat_l.var))
        blurry = var < 35.0
        tiny = pixels < (640 * 480)
        extreme_aspect = aspect > 3.5

        # sample central band color variance
        y0 = int(h * 0.45)
        y1 = int(h * 0.55)
        band = img.crop((0, y0, w, y1))
        stat_band = ImageStat.Stat(band)
        band_var = sum(stat_band.var) / max(1, len(stat_band.var))
        low_band_detail = band_var < 25.0

        # Heuristic decisions
        if tiny and blurry and extreme_aspect:
            return {"classification": "fake", "confidence": 85, "reason": "tiny+blurry+extreme_aspect"}
        if blurry and low_band_detail:
            return {"classification": "flag", "confidence": 65, "reason": "blurry+low_band_detail"}
        return {"classification": "real", "confidence": 80, "reason": "normal_variance"}
    except Exception as e:
        return {"classification": "flag", "confidence": 50, "reason": f"error:{str(e)[:60]}"}

