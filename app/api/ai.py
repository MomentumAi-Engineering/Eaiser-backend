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
    env = os.getenv("ENV", "development").lower()
    try:
        timeout = int(os.getenv("AI_TIMEOUT", "15"))
    except ValueError:
        timeout = 15
    # In production, give AI a bit more time before falling back
    if env == "production":
        timeout = max(timeout, 20)
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
                max_dim = 1024
                if max(w, h) > max_dim:
                    scale = max_dim / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=70, optimize=True)
                content = buf.getvalue()
            except Exception:
                pass

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            "Analyze the uploaded image. Identify visible issues and risks. "
            "REALITY CHECK: If the image is a CARTOON, VIDEO GAME, AI GENERATED, or FAKE, start description with 'Fake/Cartoon Image.' "
            "If the image shows a CONTROLLED FIRE (bonfire, candle, festival, stove), explicitly label it as 'Controlled Fire' - NOT a hazard. "
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
            (os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash"), max(10, timeout - 2)),
            ("gemini-1.5-pro", max(10, timeout - 3)), 
            ("gemini-2.0-flash", max(10, timeout - 4)),
            ("gemini-1.5-flash-8b", max(10, timeout - 6)),
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
            # Build local heuristic fallback instead of timeout label
            try:
                probe = Image.open(BytesIO(content)).convert("RGB")
                w, h = probe.size
                # sample central horizontal band
                y0 = int(h * 0.45)
                y1 = int(h * 0.55)
                band = probe.crop((0, y0, w, y1))
                # count brown-ish pixels (simple heuristic)
                brown = 0
                total = (y1 - y0) * w
                px = band.load()
                for x in range(0, w, max(1, w // 512)):
                    for y in range(0, (y1 - y0), max(1, (y1 - y0) // 64)):
                        r, g, b = px[x, y]
                        if r > 90 and g > 60 and b < 90 and (r - b) > 25:
                            brown += 1
                brown_ratio = brown / max(1, (w // max(1, w // 512)) * ((y1 - y0) // max(1, (y1 - y0) // 64)))
            except Exception:
                brown_ratio = 0.0

            issue_type_fb = "other"
            confidence_fb = 30
            labels_fb = []
            if brown_ratio > 0.25:
                issue_type_fb = "tree_fallen"
                confidence_fb = 75
                labels_fb.append("fallen-tree")
            else:
                labels_fb.append("analysis-fallback")

            description = "AI Analysis: Fallback heuristic used."
            return JSONResponse(content={
                "status": "timeout_fallback",
                "description": description,
                "issues": [],
                "labels": labels_fb,
                "confidence": confidence_fb,
                "issue_type": issue_type_fb,
                "severity": "medium" if issue_type_fb != "fire" else "high"
            })

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
        danger_words = [
            "out of control","emergency","uncontrolled","explosion","collapse",
            "wildfire","house fire","building fire","spread","spreading","structure","sirens"
        ]
        controlled_fire = ["campfire","bonfire","bon fire","bbq","barbecue","barbeque","grill","fire pit","controlled burn","controlled fire","festival","celebration","diwali","diya","candle","incense","lamp","stove","kitchen","smoke machine","stage"]
        minor_words = ["minor","small","tiny","cosmetic","scratch","smudge","dust","stain","low","no issue","normal","benign"]
        
        is_fake = "fake" in base_text or "cartoon" in base_text or "video game" in base_text or "ai generated" in base_text
        
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
        if is_fake:
            confidence_estimate = 0
        elif not issues and not has_danger:
            confidence_estimate = 10
        elif has_danger or (issues and not has_controlled):
            confidence_estimate = max(75, min(95, int(60 + clarity_score / 2)))
        elif has_controlled and not has_danger:
            confidence_estimate = 30
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

        # Add explicit AI Analysis statement in description
        try:
            first_line = (description.splitlines()[0] if description else "").strip()
            if not issues and not has_danger:
                description = f"AI Analysis: No public issue found. Scene: {first_line}" if first_line else "AI Analysis: No public issue found."
            else:
                if not description.lower().startswith("ai analysis:"):
                    description = f"AI Analysis: {first_line}\n{description}" if first_line else f"AI Analysis: {description}"
        except Exception:
            pass

        t = base_text
        issue_type = "other"
        animal_tokens = ["animal","deer","boar","hog","dog","cow","cat","wildlife","goat","pig"]
        accident_tokens = ["accident","collision","crash","hit","struck","run over","under car","under vehicle","roadkill"]
        has_animal = any(w in t for w in animal_tokens)
        has_accident = any(w in t for w in accident_tokens)
        if has_animal and has_accident and not any(w in t for w in ["fire","flame","burning","smoke"]):
            issue_type = "animal_accident"
        elif any(w in t for w in ["wildfire","house fire","building fire","uncontrolled fire"]):
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
