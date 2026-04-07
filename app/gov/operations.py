from fastapi import APIRouter, HTTPException, Depends, Body
from services.mongodb_service import get_db
from core.auth import get_current_user
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import logging
import httpx
import asyncio

router = APIRouter(
    prefix="/gov/v2",
    tags=["Gov Portal E2E Operations"]
)

logger = logging.getLogger(__name__)

from services.push_notification_service import trigger_push_notification

# --- Reports & Operations ---

@router.patch("/reports/{report_id}/status")
async def update_report_status(
    report_id: str,
    status: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    """
    Workable: Update the status of a specific municipal report.
    """
    db = await get_db()
    
    # Validation
    allowed_statuses = ["reported", "approved", "in-progress", "resolved", "rejected"]
    if status.lower() not in allowed_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {allowed_statuses}")

    # Use 'issues' collection as the source of truth
    update_data = {
        "status": status.lower(),
        "updated_at": datetime.utcnow(),
        "updated_by": current_user.get("sub")
    }

    result = await db["issues"].update_one(
        {"_id": ObjectId(report_id) if ObjectId.is_valid(report_id) else report_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        # Try finding by issue_id string
        result = await db["issues"].update_one(
            {"issue_id": report_id},
            {"$set": update_data}
        )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")

    # ⚡ LIVE PUSH NOTIFICATION to reporter
    try:
        issue = await db["issues"].find_one({
            "$or": [
                {"_id": ObjectId(report_id) if ObjectId.is_valid(report_id) else report_id},
                {"issue_id": report_id}
            ]
        })
        
        if issue and issue.get("user_email"):
            reporter_email = issue.get("user_email")
            reporter = await db["users"].find_one({"email": reporter_email})
            if reporter and reporter.get("push_token"):
                trigger_push_notification(
                    push_token=reporter["push_token"],
                    title=f"Report Update 🚀",
                    body=f"Your report '{issue.get('issue_type', 'item')}' status is now: {status.upper()}",
                    data={"issue_id": str(issue["_id"]), "status": status.lower()}
                )
    except Exception as e:
        logger.error(f"Failed to trigger push for status change: {e}")

    return {"success": True, "new_status": status, "report_id": report_id}

# --- Messaging & Notifications ---

@router.post("/set-expo-push-token")
async def save_push_token(
    push_token: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    db = await get_db()
    user_email = current_user.get("email", current_user.get("sub"))
    
    await db["gov_users"].update_one(
        {"email": user_email},
        {"$set": {"push_token": push_token}}
    )
    return {"success": True}


@router.post("/message")
async def send_municipal_message(
    recipient_email: str = Body(..., embed=True),
    message_content: str = Body(..., embed=True),
    channel: str = Body("in-app", embed=True),
    reply_to_text: str = Body(None, embed=True),
    current_user: dict = Depends(get_current_user)
):
    """
    Workable: Send a message to/from a citizen/crew via their chosen channel.
    """
    db = await get_db()
    
    sender = current_user.get("email", current_user.get("sub", "unknown"))
    is_staff = current_user.get("role") in ["admin", "staff", "superadmin", "super_admin", "operations", "ops_manager"]
    
    # Very important: thread_id represents the "External/Field" person's email, to group the chat.
    thread_id = recipient_email if is_staff else sender
    
    # Log the outbound communication
    communication = {
        "sender": sender,
        "recipient": recipient_email,
        "thread_id": thread_id,
        "content": message_content,
        "reply_to_text": reply_to_text,
        "channel": channel.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        "status": "delivered",
        "is_staff": is_staff
    }
    
    await db["gov_communications"].insert_one(communication)
    
    # ⚡ EXPO PUSH NOTIFICATION (If recipient is on mobile and has registered a push token)
    if is_staff:
        # Meaning a Command Center staff is sending to the field worker.
        target_user = await db["gov_users"].find_one({"email": recipient_email})
        if target_user and target_user.get("push_token"):
            push_payload = {
                "to": target_user["push_token"],
                "sound": "default",
                "title": f"EAiSER Command Center",
                "body": message_content,
                "data": {"type": "message", "sender": sender}
            }
            try:
                # Fire and forget asynchronously
                asyncio.create_task(send_push(push_payload))
            except Exception as e:
                logger.error(f"Error kicking off push notification: {e}")

    return {
        "success": True, 
        "message": f"Message delivered successfully",
        "sent_at": communication["timestamp"]
    }

async def send_push(payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post("https://exp.host/--/api/v2/push/send", json=payload)
        except Exception as e:
            logger.error(f"Expo Push failed: {e}")

@router.get("/messages/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    """
    Get unique chat threads. For staff, returns all. For field crew, returns only their thread.
    """
    db = await get_db()
    is_staff = current_user.get("role") in ["admin", "staff", "superadmin", "super_admin", "operations", "ops_manager"]
    user_email = current_user.get("email")
    
    match_stage = {} if is_staff else {"thread_id": user_email}
    
    # Group by thread_id to find distinct conversations
    pipeline = [
        {"$match": match_stage},
        {"$sort": {"timestamp": -1}}, # newest first
        {"$group": {
            "_id": "$thread_id",
            "last_message": {"$first": "$content"},
            "timestamp": {"$first": "$timestamp"},
            "channel": {"$first": "$channel"}
        }},
        {"$sort": {"timestamp": -1}}
    ]
    cursor = db["gov_communications"].aggregate(pipeline)
    results = await cursor.to_list(length=50)
    
    conversations = []
    for r in results:
        # Determine channel styling
        ch = r.get("channel", "IN-APP").upper()
        if ch == "SMS": color = "text-blue-400 bg-blue-500/10 border-blue-500/20"
        elif ch == "EMAIL": color = "text-green-400 bg-green-500/10 border-green-500/20"
        elif ch == "PUSH": color = "text-purple-400 bg-purple-500/10 border-purple-500/20"
        else: color = "text-[#D4A017] bg-[#D4A017]/10 border-[#D4A017]/20"
        
        # Calculate time string
        try:
            ts = r.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                mins = int((datetime.utcnow() - dt).total_seconds() / 60)
                if mins < 60: time_str = f"{mins} min ago" if mins > 0 else "Just now"
                elif mins < 1440: time_str = f"{mins//60}h ago"
                else: time_str = f"{mins//1440}d ago"
            else:
                time_str = "Recently"
        except:
            time_str = "Recently"
            
        # Handle cases where thread_id is missing, fallback to _id
        tid = r["_id"] or "Unknown"
            
        conversations.append({
            "email": tid,
            "name": tid.split('@')[0].title().replace('.', ' '),
            "subject": r.get("last_message", "")[:40] + "..." if len(r.get("last_message", "")) > 40 else r.get("last_message", ""),
            "time": time_str,
            "channel": ch,
            "channelColor": color
        })
        
    return {"conversations": conversations}

@router.get("/messages/history/{email}")
async def get_message_history(email: str, current_user: dict = Depends(get_current_user)):
    """
    Get full chat history for a specific thread_id (email).
    """
    db = await get_db()
    # Support backward compatibility by checking either thread_id matches or recipient matches
    cursor = db["gov_communications"].find({
        "$or": [{"thread_id": email}, {"recipient": email, "thread_id": {"$exists": False}}]
    }).sort("timestamp", 1)
    
    results = await cursor.to_list(length=100)
    
    messages = []
    for r in results:
        # Time formatting
        try:
            ts = r.get("timestamp")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                
                # Format to actual time instead of "X min ago" for better reading
                hour = dt.hour
                ampm = "AM" if hour < 12 else "PM"
                formatted_hour = hour % 12 or 12
                time_str = f"{formatted_hour}:{dt.minute:02d} {ampm}"
            else:
                time_str = ""
        except:
            time_str = ""
            
        is_staff = r.get("is_staff", False)
        sender_name = "Staff" if is_staff else r.get("sender", email).split('@')[0].title().replace('.', ' ')
        
        messages.append({
            "sender": sender_name,
            "text": r.get("content"),
            "reply_to_text": r.get("reply_to_text"),
            "time": time_str,
            "isStaff": is_staff,
            "status": r.get("status", "read")
        })
        
    return {"messages": messages}

# --- Crew Management ---

@router.get("/crew")
async def get_gov_crew(current_user: dict = Depends(get_current_user)):
    """
    Workable: Fetch all field crew members and their current status.
    """
    db = await get_db()
    
    # In a real app, this would be a separate 'crew' collection
    # For now, we return mock data that matches the frontend's High-Fidelity expectation
    crew = [
        { "id": "1", "name": "Tom Bradley", "department": "Public Works", "status": "Available", "performance": { "resolved": 28, "avg_days": 1.2, "stars": 4.7 } },
        { "id": "2", "name": "Maria Garcia", "department": "Public Works", "status": "On Site", "task": "Fallen tree — Music Row", "performance": { "resolved": 34, "avg_days": 0.9, "stars": 4.9 } },
        { "id": "3", "name": "Kevin Nguyen", "department": "Public Works", "status": "En Route", "task": "Street light — Charlotte Pike", "performance": { "resolved": 22, "avg_days": 1.5, "stars": 4.3 } },
        { "id": "4", "name": "Rachel Brown", "department": "Sanitation", "status": "On Site", "task": "Illegal dumping — Dickerson Pike", "performance": { "resolved": 19, "avg_days": 2.1, "stars": 4.1 } }
    ]
    
    # In 'workable' mode, we'd store these in MongoDB
    return crew
