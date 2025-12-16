from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter()

@router.post("/ai/fake-detect")
async def fake_detect(image: UploadFile = File(...), model: Optional[str] = "gemini"):
    """
    Minimal fake-detection endpoint.
    Returns: { success: bool, is_fake: bool, score: float }
    Replace the body with real logic (call Gemini or local model).
    """
    try:
        content = await image.read()
        # Placeholder: in prod call Gemini or local model
        # For now simple heuristic: if file size < 5KB => likely corrupted/fake (example)
        size = len(content)
        is_fake = size < 5000
        score = 0.9 if is_fake else 0.1
        return JSONResponse(content={"success": True, "is_fake": is_fake, "score": score})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
