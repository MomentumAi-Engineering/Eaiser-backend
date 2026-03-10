import asyncio
import json
import re
import io
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables and configure Gemini
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")

class AIServiceV3:
    """
    EAiSER V3 AI Service - Optimized for Single Image / Multi-issue analysis.
    Implements the latest Gemini 2.x/2.5 schema and Deterministic (0.0 temp) logic.
    """
    
    def __init__(self):
        self.model_name = os.getenv("GEMINI_MODEL_V3", "gemini-2.5-flash")
        self.temp = 0.0
        self._prompt_v3 = None

    async def _get_prompt_v3(self) -> str:
        """Load the V3 system instruction from disk with caching"""
        if self._prompt_v3:
            return self._prompt_v3
        
        try:
            prompt_path = Path(__file__).parent.parent / "prompts" / "eaiser_v3_single_image.txt"
            if prompt_path.exists():
                async with asyncio.Lock(): # Simple guard
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        self._prompt_v3 = f.read()
                return self._prompt_v3
            else:
                logger.error(f"V3 Prompt file not found: {prompt_path}")
                return "Analyze this civic incident image and return a structured report."
        except Exception as e:
            logger.error(f"Error loading V3 prompt: {e}")
            return "Analyze this civic incident image and return a structured report."

    def _get_model(self):
        """Get or initialize the generative model"""
        try:
            # Note: Using system_instruction parameter for the new SDK pattern
            prompt = asyncio.run(self._get_prompt_v3()) # Synchronous access for init
            return genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=prompt
            )
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, falling back to gemini-2.0-flash: {e}")
            self.model_name = "gemini-2.0-flash"
            return genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=prompt
            )

    async def analyze_single_image(self, image_content: bytes) -> Dict[str, Any]:
        """
        The core V3 analyzer. 
        Returns the structured JSON report defined in the EAiSER V3 specification.
        """
        try:
            # 1. Prepare Image
            img = Image.open(io.BytesIO(image_content))
            # Optimization: 1024px limit for speed
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024))
            
            # 2. Setup Model
            prompt = await self._get_prompt_v3()
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=prompt
            )
            
            # 3. Generate Content
            # We use generation_config for temperature=0.0 and response_mime_type="application/json"
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    { "text": "Image 1:" },
                    img,
                    { "text": "Analyze this civic incident image and return a structured report." }
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
            
            # 4. Parse JSON
            report = json.loads(response.text)
            
            # 5. Validation Check
            required_fields = ["report_meta", "visual_observations", "scene_description", "known_issues", "unknown_issues", "ordered_issue_list"]
            for field in required_fields:
                if field not in report:
                    raise ValueError(f"Missing required field in Gemini V3 response: {field}")
            
            return {
                "success": True,
                "data": report
            }

        except Exception as e:
            logger.error(f"V3 Analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    def map_v3_to_legacy(self, v3_report: Dict[str, Any], meta_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adapts the V3 schema to the legacy schema expected by the Frontend and Builder.
        This allows the backend to upgrade without breaking the current UI.
        """
        data = v3_report.get("data", {})
        meta = data.get("report_meta", {})
        
        # Determine primary issue
        primary_issue = "Other"
        primary_tier = 2
        confidence_str = "High"
        
        ordered = data.get("ordered_issue_list", [])
        if ordered:
            primary_issue = ordered[0].get("issue", "Other")
            # Map tier_or_severity
            tier_str = ordered[0].get("tier_or_severity", "Tier 2")
            if "Tier 1" in tier_str: primary_tier = 1
            elif "Tier 3" in tier_str or "Tier 4" in tier_str: primary_tier = 3
        
        # Find confidence from known_issues or unknown_issues if possible
        known = data.get("known_issues", [])
        unknown = data.get("unknown_issues", [])
        
        for k in known:
            if k.get("issue") == primary_issue:
                confidence_str = k.get("confidence", "High")
                break
        else:
            for u in unknown:
                if u.get("issue") == primary_issue:
                    confidence_str = u.get("confidence", "Low")
                    break
        
        conf_val = {"High": 95, "Medium": 75, "Low": 45}.get(confidence_str, 85)
        
        # Construct legacy object
        legacy = {
            "issue_overview": {
                "type": primary_issue,
                "category": f"Tier {primary_tier}",
                "severity": meta.get("final_priority", "Medium"),
                "summary": data.get("scene_description", ""),
                "summary_explanation": data.get("scene_description", ""),
                "confidence": conf_val
            },
            "detailed_analysis": {
                "root_causes": ", ".join(data.get("visual_observations", {}).get("image_1", [])),
                "potential_consequences_if_ignored": "Impact based on priority.",
                "public_safety_risk": meta.get("final_priority", "Medium").lower(),
                "environmental_impact": "low",
                "structural_implications": "low",
                "legal_or_regulatory_considerations": "Municipal codes apply.",
                "feedback": None
            },
            "recommended_actions": [f"Inspect {primary_issue}", "Verify priority"],
            "responsible_authorities_or_parties": [], # To be filled by service
            "available_authorities": [], # To be filled by service
            "ai_evaluation": {
                "issue_detected": len(known) > 0 or len(data.get("unknown_issues", [])) > 0,
                "detected_issue_type": primary_issue,
                "ai_confidence_percent": conf_val,
                "image_analysis": data.get("scene_description", ""),
                "rationale": f"Classified as Scenario {meta.get('scenario', 'Unknown')}"
            },
            "template_fields": {
                "oid": meta_overrides.get("report_id") if meta_overrides else "N/A",
                "timestamp": meta_overrides.get("local_time") if meta_overrides else datetime.now().strftime("%Y-%m-%d %H:%M"),
                "confidence": conf_val,
                "ai_tag": primary_issue,
                "address": meta_overrides.get("address") if meta_overrides else "Unknown",
                "priority": meta.get("final_priority", "Medium"),
                "image_filename": meta_overrides.get("image_filename") if meta_overrides else "image.jpg"
            },
            # V3 Native Data preserved for advanced UI
            "v3_data": data
        }
        
        # Add internal_review_required flag for the backend logic
        legacy["_manual_review_required"] = meta.get("internal_review_required", False)
        
        return legacy

    async def generate_report_v3(
        self,
        image_content: bytes,
        description: str,
        issue_type: str,
        severity: str,
        address: str,
        zip_code: Optional[str],
        latitude: float,
        longitude: float,
        issue_id: str,
        confidence: float,
        category: str,
        priority: str,
        decline_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Matches the signature of generate_report_optimized but uses V3 logic.
        """
        # Run V3 Analysis
        v3_result = await self.analyze_single_image(image_content)
        
        if not v3_result.get("success"):
            # Fallback to legacy if V3 fails (optional, or just raise error)
            raise ValueError(f"V3 Analysis failed: {v3_result.get('error')}")
        
        # Prepare metadata for mapping
        from datetime import datetime
        now = datetime.now()
        meta_overrides = {
            "report_id": f"v3-{now.year}-{int(now.timestamp()) % 1000000:06d}",
            "local_time": now.strftime("%Y-%m-%d %H:%M"),
            "address": address or "Unknown",
            "image_filename": f"V3_IMG_{now.strftime('%Y%m%d_%H%M')}.jpg"
        }
        
        # Map to Legacy Schema for Frontend support
        legacy_report = self.map_v3_to_legacy(v3_result, meta_overrides)
        
        # Add dynamic fields usually filled by routes
        legacy_report["issue_id"] = issue_id
        
        return legacy_report

def get_ai_service_v3() -> AIServiceV3:
    return AIServiceV3()
