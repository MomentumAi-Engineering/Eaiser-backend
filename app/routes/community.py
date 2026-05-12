from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson.objectid import ObjectId
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer

import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load env
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

router = APIRouter()
logger = logging.getLogger(__name__)

# ─── Auth (same pattern as routes/auth.py) ───
try:
    from utils.security import SECRET_KEY, ALGORITHM
except ImportError:
    SECRET_KEY = os.getenv("SECRET_KEY", "eaiser-secret-key-2025")
    ALGORITHM = os.getenv("ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return {"sub": email, "id": payload.get("id"), "role": payload.get("role")}
    except JWTError:
        raise credentials_exception

# ─── Database ───
try:
    from services.mongodb_service import get_db
except ImportError:
    from app.services.mongodb_service import get_db


# ═══════════════════════════════════════
# COMMUNITY POSTS CRUD
# ═══════════════════════════════════════

@router.get("/posts", response_model=List[Dict[str, Any]])
async def get_community_posts(
    skip: int = 0,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Fetch paginated community posts, newest first."""
    try:
        db = await get_db()
        cursor = db["community_posts"].find({}).sort("created_at", -1).skip(skip).limit(limit)
        posts = await cursor.to_list(length=limit)

        # Get current user's ID for like status
        user_email = current_user.get("sub", "")

        for post in posts:
            post["id"] = str(post.pop("_id"))
            # Add relative time
            if "created_at" in post and isinstance(post["created_at"], datetime):
                delta = datetime.utcnow() - post["created_at"]
                if delta.total_seconds() < 60:
                    post["time"] = "Just now"
                elif delta.total_seconds() < 3600:
                    post["time"] = f"{int(delta.total_seconds() // 60)}m ago"
                elif delta.total_seconds() < 86400:
                    post["time"] = f"{int(delta.total_seconds() // 3600)}h ago"
                else:
                    post["time"] = f"{int(delta.days)}d ago"
            else:
                post["time"] = "Just now"

            # Check if current user liked this post
            likes = post.get("likes", [])
            post["liked"] = user_email in likes

            # Add initials to user if missing
            if "user" in post and isinstance(post["user"], dict):
                name = post["user"].get("name", "?")
                if not post["user"].get("initials"):
                    post["user"]["initials"] = name[0].upper() if name else "?"

        return posts
    except Exception as e:
        logger.error(f"Error fetching posts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch community posts")


@router.post("/posts")
async def create_community_post(
    post_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Create a new community post."""
    try:
        db = await get_db()

        # Fetch full user profile for rich post data
        user = await db["users"].find_one({"email": current_user["sub"]})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        first_name = user.get("first_name", "")
        last_name = user.get("last_name", "")
        display_name = f"{first_name} {last_name}".strip() or user.get("name", "User")

        new_post = {
            "user": {
                "id": str(user["_id"]),
                "name": display_name,
                "handle": user.get("username") or user.get("email", "user@x").split("@")[0].lower(),
                "avatar": user.get("avatar", None),
                "initials": display_name[0].upper() if display_name else "?",
                "badge": "Member"
            },
            "content": post_data.get("content", ""),
            "image": post_data.get("image", None),
            "aiReply": post_data.get("aiReply", None),
            "likes": [],
            "comments": [],
            "shares": 0,
            "created_at": datetime.utcnow()
        }

        result = await db["community_posts"].insert_one(new_post)

        # Return the created post
        created_post = await db["community_posts"].find_one({"_id": result.inserted_id})
        created_post["id"] = str(created_post.pop("_id"))
        created_post["time"] = "Just now"
        created_post["liked"] = False

        return created_post
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating post: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create post")


@router.post("/posts/{post_id}/like")
async def toggle_like(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Toggle like on a community post."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        post = await db["community_posts"].find_one({"_id": ObjectId(post_id)})
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        likes = post.get("likes", [])
        if user_email in likes:
            likes.remove(user_email)
            liked = False
        else:
            likes.append(user_email)
            liked = True

        await db["community_posts"].update_one({"_id": ObjectId(post_id)}, {"$set": {"likes": likes}})
        return {"liked": liked, "likes_count": len(likes)}
    except Exception as e:
        logger.error(f"Error toggling like: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle like")


@router.post("/posts/{post_id}/comment")
async def add_comment(
    post_id: str,
    comment_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Add a comment or reply to a community post."""
    try:
        db = await get_db()

        post = await db["community_posts"].find_one({"_id": ObjectId(post_id)})
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Fetch user profile
        user = await db["users"].find_one({"email": current_user["sub"]})
        display_name = "User"
        user_handle = "user"
        user_avatar = None
        if user:
            first = user.get("first_name", "")
            last = user.get("last_name", "")
            display_name = f"{first} {last}".strip() or user.get("name", "User")
            user_handle = user.get("username") or user.get("email", "user@x").split("@")[0].lower()
            user_avatar = user.get("avatar")

        parent_id = comment_data.get("parentId")

        new_comment = {
            "id": str(ObjectId()),
            "user": display_name,
            "user_handle": user_handle,
            "user_avatar": user_avatar,
            "text": comment_data.get("text", ""),
            "time": "Just now",
            "likes": 0,
            "liked": False,
            "created_at": datetime.utcnow(),
            "replies": []
        }

        comments = post.get("comments", [])

        if not parent_id:
            comments.append(new_comment)
        else:
            for c in comments:
                if c.get("id") == parent_id:
                    if "replies" not in c:
                        c["replies"] = []
                    c["replies"].append(new_comment)
                    break

        await db["community_posts"].update_one({"_id": ObjectId(post_id)}, {"$set": {"comments": comments}})
        return new_comment
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add comment")


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a community post (only by the post owner)."""
    try:
        db = await get_db()

        post = await db["community_posts"].find_one({"_id": ObjectId(post_id)})
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Check if post belongs to user
        post_user_id = post.get("user", {}).get("id", "")
        user = await db["users"].find_one({"email": current_user["sub"]})
        current_user_id = str(user["_id"]) if user else ""

        if post_user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this post")

        await db["community_posts"].delete_one({"_id": ObjectId(post_id)})
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting post: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete post")


# ═══════════════════════════════════════
# AI CHAT (Community Bot)
# ═══════════════════════════════════════

@router.post("/chat")
async def community_ai_chat(
    data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """AI-powered community chat bot using Gemini."""
    try:
        message = data.get("message", "").strip()
        if not message:
            return {"reply": "Hey! Tag me with a question about civic issues and I'll help out! 🏙️"}

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            return {"reply": "I'm temporarily offline. Our AI service will be back shortly! 🔧"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""You are EAiSER AI — the official community assistant for the EAiSER civic-tech platform.

EAiSER helps residents report municipal issues like potholes, broken streetlights, flooding, garbage, graffiti, fallen trees, traffic signal damage, water leakage, abandoned vehicles, and more. Reports are AI-analyzed, then routed to local authorities for resolution.

RULES:
- Keep responses concise (2-4 sentences max)
- Be friendly, professional, and encouraging
- If asked about civic issues, give practical advice
- If asked about EAiSER features, explain clearly
- If the user shares a concern, acknowledge it and suggest they submit a report
- Use relevant emojis sparingly (1-2 per response)
- Never make up statistics or false claims
- If unsure, suggest the user submit a report for official tracking

User message: {message}

Your helpful response:"""

        response = model.generate_content(prompt)
        reply_text = response.text.strip() if response and response.text else "I'm here to help with civic reporting! Try submitting a report through the app. 📋"

        return {"reply": reply_text}
    except Exception as e:
        logger.error(f"Community AI chat error: {str(e)}")
        return {"reply": "I'm processing a lot of civic data right now. Please try again in a moment! 🔄"}
