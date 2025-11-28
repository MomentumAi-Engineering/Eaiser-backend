from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import asyncio
from io import BytesIO
from PIL import Image, ImageStat
import google.generativeai as genai

router = APIRouter()

@router.post("/ai/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
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
        if len(content) > 2_000_000:
            try:
                img = Image.open(BytesIO(content))
                img = img.convert("RGB")
                w, h = img.size
                max_dim = 1280
                if max(w, h) > max_dim:
                    scale = max_dim / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=80, optimize=True)
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
            description = "Analysis temporarily unavailable. Please try Analyze Again."
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

        base_text = (description or "").lower() + " " + " ".join([str(x).lower() for x in labels])
        danger_words = ["hazard","danger","out of control","emergency","injury","uncontrolled","explosion","collapse","severe","major","wildfire","accident","collision","leak","burst"]
        controlled_fire = ["campfire","bonfire","bon fire","bbq","barbecue","barbeque","grill","fire pit","controlled burn","festival","celebration","diwali","diya","candle","incense","lamp","stove","kitchen","smoke machine","stage"]
        minor_words = ["minor","small","tiny","cosmetic","scratch","smudge","dust","stain","low","no issue","normal","benign"]
        has_danger = any(w in base_text for w in danger_words)
        has_controlled = any(w in base_text for w in controlled_fire)
        has_minor = any(w in base_text for w in minor_words)
        clarity_score = 50
        try:
            probe = Image.open(BytesIO(content)).convert("L")
            w, h = probe.size
            pixels = w * h
            hi_res = pixels >= (1000 * 800)
            stat = ImageStat.Stat(probe)
            var = sum(stat.var) / max(1, len(stat.var))
            blurry = var < 50.0
            clarity_score = 70 if hi_res else 40
            if blurry:
                clarity_score -= 20
            clarity_score = max(0, min(100, clarity_score))
        except Exception:
            clarity_score = 50
        confidence_estimate = 20
        if not issues and not has_danger:
            confidence_estimate = 10
        elif has_danger or (issues and not has_controlled):
            confidence_estimate = max(75, min(95, int(60 + clarity_score / 2)))
        elif has_controlled and not has_danger:
            confidence_estimate = 45
        elif has_minor and not has_danger:
            confidence_estimate = max(75, min(90, int(55 + clarity_score / 2)))
        # Boost confidence for explicit target issue tokens
        target_issue_tokens = [
            "pothole", "road damage", "road_damage",
            "fire", "wildfire",
            "dead animal", "animal carcass", "animal_carcass",
            "garbage", "trash", "waste",
            "flood", "waterlogging",
            "tree fallen", "fallen tree", "tree_fallen",
            "public toilet", "public_toilet_issue"
        ]
        has_target_issue = any(t in base_text for t in target_issue_tokens)
        if has_target_issue and not has_controlled:
            confidence_estimate = max(confidence_estimate, max(75, min(95, int(60 + clarity_score / 2))))
        confidence_estimate = int(max(0, min(100, confidence_estimate)))

        t = base_text
        issue_type = "other"
        if any(w in t for w in ["wildfire","house fire","building fire","uncontrolled fire"]):
            issue_type = "fire"
        elif any(w in t for w in ["fire","smoke","flame","burning"]) and not has_controlled:
            issue_type = "fire"
        elif any(w in t for w in ["flood","waterlogging","inundation"]):
            issue_type = "flood"
        elif any(w in t for w in ["leak","burst","pipeline","water leak","pipe leak","water leakage"]):
            issue_type = "water_leakage"
        elif any(w in t for w in ["pothole","road damage","crack","asphalt","hole","road_damage"]):
            issue_type = "road_damage"
        elif any(w in t for w in ["fallen tree","tree fallen","tree down","branch fallen","tree_fallen"]):
            issue_type = "tree_fallen"
        elif any(w in t for w in ["dead animal","carcass","roadkill","animal carcass","animal_carcass"]):
            issue_type = "dead_animal"
        elif any(w in t for w in ["garbage","trash","waste","dump","litter"]):
            issue_type = "garbage"
        elif any(w in t for w in ["public toilet","washroom","restroom","sanitation","urinal","toilet"]):
            issue_type = "public_toilet_issue"
        elif any(w in t for w in ["streetlight","street light","lamp post","light pole","bulb broken","broken light"]):
            issue_type = "broken_streetlight"
        elif any(w in t for w in ["graffiti","spray paint","tagging","defaced"]):
            issue_type = "graffiti"
        elif any(w in t for w in ["vandalism","smashed","broken window","defaced property"]):
            issue_type = "vandalism"
        elif any(w in t for w in ["open drain","open manhole","uncovered drain"]):
            issue_type = "open_drain"
        elif any(w in t for w in ["blocked drain","clogged","sewer blockage","drain blocked"]):
            issue_type = "blocked_drain"
        elif any(w in t for w in ["signal malfunction","traffic light","signal not working","broken signal"]):
            issue_type = "signal_malfunction"
        elif any(w in t for w in ["street vendor","encroachment","hawker","footpath occupied"]):
            issue_type = "street_vendor_encroachment"
        elif any(w in t for w in ["abandoned vehicle","junk car","derelict","left vehicle"]):
            issue_type = "abandoned_vehicle"
        elif any(w in t for w in ["vacant lot","empty plot","illegal dumping"]):
            issue_type = "vacant_lot_issue"
        elif any(w in t for w in ["noise pollution","loud music","noise","honking excessive"]):
            issue_type = "noise_pollution"
        elif any(w in t for w in ["air pollution","smog","emission","smoke"]):
            issue_type = "air_pollution"
        elif any(w in t for w in ["illegal construction","unauthorized building","unapproved construction"]):
            issue_type = "illegal_construction"
        elif any(w in t for w in ["stray animals","stray dog","stray cow"]):
            issue_type = "stray_animals"
        elif any(w in t for w in ["animal injury","injured animal"]):
            issue_type = "animal_injury"
        elif any(w in t for w in ["animal accident","animal crash","hit animal"]):
            issue_type = "animal_accident"
        elif any(w in t for w in ["wildlife hit","deer hit"]):
            issue_type = "wildlife_hit"
        elif any(w in t for w in ["animal on road"]):
            issue_type = "animal_on_road"

        allowed_types = {
            "pothole","road_damage","broken_streetlight","graffiti","garbage","vandalism","open_drain","blocked_drain","flood","fire","illegal_construction","tree_fallen","public_toilet_issue","stray_animals","dead_animal","animal_carcass","animal_injury","animal_accident","wildlife_hit","animal_on_road","animal_crash","noise_pollution","air_pollution","water_leakage","street_vendor_encroachment","signal_malfunction","waterlogging","abandoned_vehicle","vacant_lot_issue","other","unknown"
        }
        if issue_type not in allowed_types:
            issue_type = "unknown"
            confidence_estimate = min(confidence_estimate, 40)
            if not has_danger:
                description = "AI Analysis: I did not find any public issue."

        severity = "medium"
        if has_danger:
            severity = "high"
        else:
            if issue_type in ["garbage","public_toilet_issue","graffiti","vandalism","open_drain","blocked_drain","noise_pollution","street_vendor_encroachment"]:
                severity = "low"
            elif issue_type in ["road_damage","tree_fallen","dead_animal","water_leakage","flood","signal_malfunction","abandoned_vehicle","vacant_lot_issue"]:
                severity = "medium"
            elif issue_type == "fire":
                severity = "high" if not has_controlled else "low"

        # Infer issue_type from description + labels tokens
        t = base_text
        issue_type = "other"
        if any(w in t for w in ["wildfire","house fire","building fire","uncontrolled fire"]):
            issue_type = "fire"
        elif any(w in t for w in ["fire","smoke","flame","burning"]) and not has_controlled:
            issue_type = "fire"
        elif any(w in t for w in ["flood","waterlogging","inundation"]):
            issue_type = "flood"
        elif any(w in t for w in ["leak","burst","pipeline","water leak","pipe leak"]):
            issue_type = "leak"
        elif any(w in t for w in ["pothole","road damage","crack","asphalt","hole","road_damage"]):
            issue_type = "road_damage"
        elif any(w in t for w in ["fallen tree","tree fallen","tree down","branch fallen","tree_fallen"]):
            issue_type = "tree_fallen"
        elif any(w in t for w in ["dead animal","carcass","roadkill","animal carcass","animal_carcass"]):
            issue_type = "dead_animal"
        elif any(w in t for w in ["garbage","trash","waste","dump","litter"]):
            issue_type = "garbage"
        elif any(w in t for w in ["public toilet","washroom","restroom","sanitation","urinal","toilet"]):
            issue_type = "public_toilet_issue"
        elif any(w in t for w in ["streetlight","street light","lamp post","light pole","bulb broken","broken light"]):
            issue_type = "broken_streetlight"

        # Heuristic severity based on danger and issue type
        severity = "medium"
        if has_danger:
            severity = "high"
        else:
            if issue_type in ["garbage","public_toilet_issue"]:
                severity = "low"
            elif issue_type in ["road_damage","tree_fallen","dead_animal","leak","flood"]:
                severity = "medium"
            elif issue_type == "fire":
                severity = "high" if not has_controlled else "low"

        return JSONResponse(
            content={
                "status": "success",
                "description": description,
                "issues": issues,
                "labels": labels,
                "confidence": confidence_estimate,
                "issue_type": issue_type,
                "severity": severity,
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
async def analyze_image_alias(image: UploadFile = File(...)):
    return await analyze_image(image=image)