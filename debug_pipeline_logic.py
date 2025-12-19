import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

# Load env
env_path = Path(__file__).parent / "app" / ".env"
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGODB_NAME")

async def simulate_endpoint_logic():
    print(f"Connecting to DB...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db.issues
    
    print("\n--- SIMULATING get_pending_reviews LOGIC ---")
    cursor = collection.find({}).sort("timestamp", -1).limit(20)
    recent_issues = await cursor.to_list(length=20)
    
    final_reviews = []
    
    for issue in recent_issues:
        sid = str(issue["_id"])
        status = issue.get("status", "unknown")
        
        should_show = False
        reason = []
        
        # Logic from endpoint
        if status == "needs_review":
            should_show = True
            reason.append("status is needs_review")
        
        elif status == "screened_out" or issue.get("dispatch_decision") == "reject":
            should_show = True
            reason.append("screened_out or rejected")
            
        elif status in ["submitted", "pending", "completed", "accepted"]: # Added accepted just in case
            # manual extraction
            conf_values = []
            
            def parse_conf(val):
                if val is None: return None
                try:
                    s = str(val).replace("%", "").strip()
                    return float(s)
                except: return None

            c1 = parse_conf(issue.get("confidence"))
            c2 = parse_conf(issue.get("report", {}).get("issue_overview", {}).get("confidence"))
            c3 = parse_conf(issue.get("report", {}).get("template_fields", {}).get("confidence"))
            
            if c1 is not None: conf_values.append(c1)
            if c2 is not None: conf_values.append(c2)
            if c3 is not None: conf_values.append(c3)
            
            effective_conf = min(conf_values) if conf_values else 0
            
            desc = str(issue.get("description") or "").lower()
            ai_summary = str(issue.get("report", {}).get("issue_overview", {}).get("summary_explanation") or "").lower()
            combined_text = desc + " " + ai_summary
            is_fake_text = any(x in combined_text for x in ["fake", "cartoon", "ai generate"])
            
            if effective_conf < 70:
                should_show = True
                reason.append(f"Low confidence: {effective_conf}")
            
            if is_fake_text:
                should_show = True
                reason.append("Fake text detected")
                
            if issue.get("issue_type") == "unknown":
                should_show = True
                reason.append("Issue type unknown")

        print(f"ID: {sid} | Status: {status} | Show: {should_show} | Reasons: {', '.join(reason)}")
        
        if should_show:
            final_reviews.append(issue)

    print(f"\nTotal items that WOULD return: {len(final_reviews)}")

if __name__ == "__main__":
    asyncio.run(simulate_endpoint_logic())
