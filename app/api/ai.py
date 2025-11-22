from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import os
import asyncio
from io import BytesIO
from PIL import Image
import google.generativeai as genai

router = APIRouter()

@router.post("/ai/analyze-image")
async def analyze_image(image: UploadFile = File(...), fast: bool = Query(False)):
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        timeout = int(os.getenv("AI_TIMEOUT", "15"))
    except ValueError:
        timeout = 15
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    try:
        content = await image.read()

        # compress large images to reduce latency
        if fast or len(content) > 2_000_000:
            try:
                img = Image.open(BytesIO(content))
                img = img.convert("RGB")
                w, h = img.size
                max_dim = 1024 if fast else 1280
                if max(w, h) > max_dim:
                    scale = max_dim / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=70 if fast else 80, optimize=True)
                content = buf.getvalue()
            except Exception:
                pass

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            "Analyze the uploaded image. Identify visible issues and risks. "
            "Return a concise issue description and list of detected issues as bullets. "
            "Also include short scene labels describing what is happening."
        )
        img_part = {"mime_type": "image/jpeg", "data": content}

        # helper to call model with guarded timeout
        async def run_once(cur_model_name: str, cur_timeout: int):
            m = genai.GenerativeModel(cur_model_name)
            return await asyncio.to_thread(
                m.generate_content,
                [prompt, img_part],
                request_options={"timeout": cur_timeout}
            )

        text = ""
        attempts = [
            (model_name, timeout),
            (model_name, max(10, timeout - 5)),
            (os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash"), max(10, timeout - 10)),
        ]
        for mdl, tmo in attempts:
            try:
                resp = await asyncio.wait_for(run_once(mdl, tmo), timeout=tmo + 5)
                text = resp.text or ""
                if text:
                    break
            except Exception:
                continue
        if not text:
            description = "Analysis temporarily unavailable. Please try Analyze Again or use Fast Mode."
            return JSONResponse(content={"status": "timeout_fallback", "description": description, "issues": [], "labels": ["analysis-timeout"]})

        description = text.strip()
        issues = []
        labels = []
        for line in description.splitlines():
            l = line.strip(" -*•")
            if not l:
                continue
            if any(k in l.lower() for k in ["issue", "risk", "problem", "damage", "hazard"]):
                issues.append(l)
            elif len(labels) < 8:
                labels.append(l)

        return JSONResponse(
            content={
                "status": "success",
                "description": description,
                "issues": issues,
                "labels": labels,
            }
        )
    except Exception as e:
        msg = str(e) if e else ""
        if "Deadline" in msg or "deadline" in msg or "504" in msg:
            return JSONResponse(
                content={
                    "status": "error_fallback",
                    "description": "Upstream AI deadline exceeded. Showing fallback summary.",
                    "issues": [],
                    "labels": ["deadline-exceeded"],
                }
            )
        raise HTTPException(status_code=500, detail=msg)

@router.post("/analyze-image")
async def analyze_image_alias(image: UploadFile = File(...), fast: bool = Query(False)):
    return await analyze_image(image=image, fast=fast)