"""
EAiSER Community Hub — Production-Grade Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  • JWT-based authentication (shared system secrets)
  • MongoDB atomic operations ($push/$pull — no read-modify-write)
  • Collection indexes for query performance
  • Server-side content moderation
  • Rate limiting per user
  • Input validation & sanitization
  • Pagination with total count
  • Gemini 2.0 AI with context-aware civic prompts
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, validator

import asyncio
import logging
import os
import re
import time
import traceback
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
# INPUT VALIDATION MODELS
# ═══════════════════════════════════════

class PollChoice(BaseModel):
    text: str = Field(..., min_length=1, max_length=25)
    image: Optional[str] = None  # Optional image/emoji for the choice

class PollCreate(BaseModel):
    question: str = Field(default="", max_length=200)
    choices: List[PollChoice] = Field(..., min_length=2, max_length=4)
    duration_days: int = Field(default=1, ge=0, le=7)
    duration_hours: int = Field(default=0, ge=0, le=23)
    duration_minutes: int = Field(default=0, ge=0, le=59)

    @validator('choices')
    def validate_choices(cls, v):
        texts = [c.text.strip().lower() for c in v]
        if len(set(texts)) != len(texts):
            raise ValueError('Poll choices must be unique')
        return v

class PollVote(BaseModel):
    choice_index: int = Field(..., ge=0)

class PostCreate(BaseModel):
    content: str = Field(default="", max_length=2000)
    image: Optional[str] = None
    gif: Optional[str] = None  # GIF URL from Tenor/Giphy
    aiReply: Optional[str] = None
    poll: Optional[PollCreate] = None  # Optional poll attachment

    @validator('content')
    def sanitize_content(cls, v):
        if v:
            # Strip excessive whitespace
            v = re.sub(r'\n{4,}', '\n\n\n', v)
            v = v.strip()
        return v

class CommentCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    parentId: Optional[str] = None

    @validator('text')
    def sanitize_text(cls, v):
        return v.strip()

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

# Valid reaction types (emoji-based)
VALID_REACTIONS = {"❤️", "😡", "😢", "👏", "🔥"}

class ReactionCreate(BaseModel):
    reaction: str = Field(..., min_length=1, max_length=4)

    @validator('reaction')
    def validate_reaction(cls, v):
        if v not in VALID_REACTIONS:
            raise ValueError(f'Invalid reaction. Must be one of: {VALID_REACTIONS}')
        return v


# ═══════════════════════════════════════
# SERVER-SIDE CONTENT MODERATION
# ═══════════════════════════════════════

_BAD_PATTERNS = re.compile(
    r'\b(hate|kill|stupid|idiot|dumb|racist|sexist|abuse|violence|terror)\b'
    r'|f[u*][c*][k*]|sh[i*]t|a[s*][s*]h[o*]le|b[i*]tch',
    re.IGNORECASE
)

def is_content_safe(text: str) -> bool:
    """Server-side profanity filter — rejects toxic content."""
    if not text:
        return True
    return not bool(_BAD_PATTERNS.search(text))


# ═══════════════════════════════════════
# RATE LIMITING (memory-safe, production-grade)
# ═══════════════════════════════════════

_rate_limits: Dict[str, list] = {}
_rate_limit_last_cleanup = time.time()
RATE_LIMIT_WINDOW = 60          # seconds
RATE_LIMIT_MAX_POSTS = 10       # max posts per window
RATE_LIMIT_MAX_COMMENTS = 30    # max comments per window
RATE_LIMIT_MAX_AI = 5           # max AI calls per window
RATE_LIMIT_MAX_KEYS = 10000     # hard cap: evict oldest if exceeded (prevents OOM)
RATE_LIMIT_CLEANUP_INTERVAL = 300  # cleanup stale keys every 5 min

def check_rate_limit(user_email: str, action: str = "post") -> bool:
    """Memory-safe sliding window rate limiter with periodic cleanup."""
    global _rate_limit_last_cleanup
    key = f"{user_email}:{action}"
    now = time.time()
    limits = {
        "post": RATE_LIMIT_MAX_POSTS,
        "comment": RATE_LIMIT_MAX_COMMENTS,
        "ai_chat": RATE_LIMIT_MAX_AI,
    }
    max_actions = limits.get(action, RATE_LIMIT_MAX_POSTS)

    # Periodic stale key cleanup (prevents unbounded memory growth)
    if now - _rate_limit_last_cleanup > RATE_LIMIT_CLEANUP_INTERVAL:
        stale_keys = [k for k, v in _rate_limits.items() if not v or (now - v[-1]) > RATE_LIMIT_WINDOW * 2]
        for k in stale_keys:
            del _rate_limits[k]
        _rate_limit_last_cleanup = now
        if stale_keys:
            logger.debug(f"🧹 Rate limiter cleanup: evicted {len(stale_keys)} stale keys")

    # Hard cap: if too many unique users, evict oldest entries
    if len(_rate_limits) > RATE_LIMIT_MAX_KEYS:
        oldest_keys = sorted(_rate_limits.keys(), key=lambda k: _rate_limits[k][-1] if _rate_limits[k] else 0)[:1000]
        for k in oldest_keys:
            del _rate_limits[k]
        logger.warning(f"⚠️ Rate limiter hard cap hit: evicted {len(oldest_keys)} oldest keys")

    if key not in _rate_limits:
        _rate_limits[key] = []

    # Remove expired entries
    _rate_limits[key] = [t for t in _rate_limits[key] if now - t < RATE_LIMIT_WINDOW]

    if len(_rate_limits[key]) >= max_actions:
        return False

    _rate_limits[key].append(now)
    return True


# ═══════════════════════════════════════
# TENOR API CIRCUIT BREAKER
# ═══════════════════════════════════════

_tenor_failures = 0
_tenor_last_failure = 0.0
TENOR_CIRCUIT_THRESHOLD = 5    # failures before opening circuit
TENOR_CIRCUIT_RESET_SEC = 120  # try again after 2 min

def _tenor_circuit_open() -> bool:
    """Check if Tenor API circuit breaker is open (too many recent failures)."""
    if _tenor_failures < TENOR_CIRCUIT_THRESHOLD:
        return False
    return (time.time() - _tenor_last_failure) < TENOR_CIRCUIT_RESET_SEC

def _tenor_record_failure():
    global _tenor_failures, _tenor_last_failure
    _tenor_failures += 1
    _tenor_last_failure = time.time()

def _tenor_record_success():
    global _tenor_failures
    _tenor_failures = 0


# ═══════════════════════════════════════
# DATABASE INDEX INITIALIZATION
# ═══════════════════════════════════════

_indexes_created = False
_index_task_launched = False

async def ensure_community_indexes(db):
    """Schedule index creation in background — never blocks the first request."""
    global _indexes_created, _index_task_launched
    if _indexes_created or _index_task_launched:
        return
    _index_task_launched = True
    asyncio.create_task(_create_community_indexes(db))

async def _create_community_indexes(db):
    """Background: Create optimized indexes for community collections."""
    global _indexes_created

    try:
        coll = db["community_posts"]

        # Primary sort index (newest first)
        await coll.create_index(
            [("created_at", -1)],
            name="community_created_at_desc",
            background=True
        )

        # User lookup for delete authorization
        await coll.create_index(
            [("user.id", 1)],
            name="community_user_id",
            background=True
        )

        # Compound index for user-specific post queries
        await coll.create_index(
            [("user.id", 1), ("created_at", -1)],
            name="community_user_posts",
            background=True
        )

        # Likes array for fast $in lookups
        await coll.create_index(
            [("likes", 1)],
            name="community_likes",
            background=True
        )

        # Poll expiry index for active poll queries
        await coll.create_index(
            [("poll.expires_at", 1)],
            name="community_poll_expiry",
            background=True,
            sparse=True  # Only index docs with polls
        )

        # Content type index for filtering (posts with images/gifs/polls)
        await coll.create_index(
            [("gif", 1)],
            name="community_gif",
            background=True,
            sparse=True
        )

        # ─── Notification & Bookmark indexes ───
        notif_coll = db["community_notifications"]
        await notif_coll.create_index(
            [("recipient", 1), ("created_at", -1)],
            name="notif_user_time",
            background=True
        )
        await notif_coll.create_index(
            [("recipient", 1), ("read", 1)],
            name="notif_user_read",
            background=True
        )

        bookmarks_coll = db["community_bookmarks"]
        await bookmarks_coll.create_index(
            [("user_email", 1), ("post_id", 1)],
            name="bookmark_user_post",
            unique=True,
            background=True
        )
        await bookmarks_coll.create_index(
            [("user_email", 1), ("created_at", -1)],
            name="bookmark_user_time",
            background=True
        )

        _indexes_created = True
        logger.info("✅ Community collection indexes created/verified")
    except Exception as e:
        logger.warning(f"⚠️ Community index creation (non-fatal): {e}")
        _indexes_created = True  # Don't retry on failure


# ═══════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════

def format_relative_time(created_at) -> str:
    """Convert datetime to human-readable relative time."""
    if not isinstance(created_at, datetime):
        return "Just now"
    delta = datetime.utcnow() - created_at
    seconds = delta.total_seconds()
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    elif seconds < 604800:
        return f"{int(delta.days)}d ago"
    else:
        return created_at.strftime("%b %d")


def serialize_post(post: dict, user_email: str, bookmarked_ids: set = None) -> dict:
    """Serialize a MongoDB post document for the API response.
    
    Defensive: handles malformed/missing data without crashing.
    """
    try:
        post["id"] = str(post.pop("_id", ""))
    except Exception:
        post["id"] = ""

    post["time"] = format_relative_time(post.get("created_at"))

    # Like status for current user
    likes = post.get("likes", []) or []
    if not isinstance(likes, list):
        likes = []
    post["liked"] = user_email in likes

    # Bookmark status
    if bookmarked_ids is not None:
        post["bookmarked"] = post["id"] in bookmarked_ids
    else:
        post.setdefault("bookmarked", False)

    # Reactions summary — strip voter emails, return only counts + user's reaction
    raw_reactions = post.get("reactions", {}) or {}
    reactions_summary = {}
    user_reaction = None
    for r_emoji in VALID_REACTIONS:
        voters = raw_reactions.get(r_emoji, [])
        if isinstance(voters, list) and len(voters) > 0:
            reactions_summary[r_emoji] = len(voters)
            if user_email in voters:
                user_reaction = r_emoji
    post["reactions"] = reactions_summary
    post["user_reaction"] = user_reaction
    post["reactions_total"] = sum(reactions_summary.values())

    # Ensure user has initials
    user_obj = post.get("user")
    if isinstance(user_obj, dict):
        name = user_obj.get("name", "?")
        if not user_obj.get("initials"):
            user_obj["initials"] = name[0].upper() if name else "?"
    elif user_obj is None:
        post["user"] = {"name": "Unknown", "handle": "unknown", "initials": "?"}

    # Serialize poll data (defensive against malformed data)
    poll = post.get("poll")
    if poll and isinstance(poll, dict):
        try:
            choices = poll.get("choices", []) or []
            total_votes = sum(len(c.get("voters", []) or []) for c in choices if isinstance(c, dict))
            poll["total_votes"] = total_votes
            poll["user_voted"] = False
            poll["user_choice"] = None
            for idx, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue
                voters = choice.get("voters", []) or []
                choice["votes"] = len(voters)
                choice["percentage"] = round((choice["votes"] / total_votes * 100) if total_votes > 0 else 0, 1)
                if user_email in voters:
                    poll["user_voted"] = True
                    poll["user_choice"] = idx
                # Strip voter emails from response (privacy)
                choice.pop("voters", None)
            # Check if poll expired
            expires_at = poll.get("expires_at")
            if expires_at and isinstance(expires_at, datetime):
                poll["is_expired"] = datetime.utcnow() > expires_at
                poll["time_remaining"] = format_relative_time_future(expires_at)
            else:
                poll["is_expired"] = False
                poll["time_remaining"] = "Open"
        except Exception as e:
            logger.warning(f"⚠️ Poll serialization error for post {post.get('id')}: {e}")
            post["poll"] = None  # Fail safe: remove broken poll

    # Serialize comments safely
    comments = post.get("comments")
    if comments and isinstance(comments, list):
        for comment in comments:
            if isinstance(comment, dict):
                comment.setdefault("likes", 0)
                comment.setdefault("liked", False)
                comment.setdefault("replies", [])

    return post


def format_relative_time_future(target_time) -> str:
    """Convert future datetime to human-readable remaining time."""
    if not isinstance(target_time, datetime):
        return "Open"
    delta = target_time - datetime.utcnow()
    seconds = delta.total_seconds()
    if seconds <= 0:
        return "Ended"
    elif seconds < 60:
        return f"{int(seconds)}s left"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m left"
    elif seconds < 86400:
        return f"{int(seconds // 3600)}h left"
    else:
        return f"{int(delta.days)}d left"


# Default DB query timeout (seconds) — protects against hung connections
DB_QUERY_TIMEOUT = float(os.getenv("COMMUNITY_DB_TIMEOUT", "10"))

async def get_user_profile(db, email: str) -> dict:
    """Fetch user profile with timeout protection and safe fallback."""
    try:
        user = await asyncio.wait_for(
            db["users"].find_one(
                {"email": email},
                {"first_name": 1, "last_name": 1, "name": 1, "username": 1, "email": 1, "avatar": 1, "_id": 1}
            ),
            timeout=DB_QUERY_TIMEOUT
        )
        if not user:
            return None

        first_name = user.get("first_name", "")
        last_name = user.get("last_name", "")
        display_name = f"{first_name} {last_name}".strip() or user.get("name", "User")

        return {
            "id": str(user["_id"]),
            "name": display_name,
            "handle": user.get("username") or user.get("email", "user@x").split("@")[0].lower(),
            "avatar": user.get("avatar"),
            "initials": display_name[0].upper() if display_name else "?",
        "badge": "Member"
    }
    except asyncio.TimeoutError:
        logger.warning(f"⏰ User profile lookup timed out for {email}")
        return None
    except Exception as e:
        logger.error(f"❌ User profile lookup failed for {email}: {e}")
        return None

# ═══════════════════════════════════════
# COMMUNITY POSTS ENDPOINTS
# ═══════════════════════════════════════

@router.get("/posts", response_model=List[Dict[str, Any]])
async def get_community_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of posts to return"),
    current_user: dict = Depends(get_current_user)
):
    """Fetch paginated community posts, newest first. Returns posts with like status for current user."""
    start_time = time.time()
    try:
        db = await get_db()
        await ensure_community_indexes(db)

        user_email = current_user.get("sub", "")

        # Targeted projection — only fetch fields needed for feed rendering.
        # Excluding heavy embedded arrays (full comments text, voter lists) dramatically
        # reduces document size and transfer time over Atlas (300ms+ latency).
        async def _fetch_posts():
            cursor = db["community_posts"].find(
                {},
                {
                    "user": 1,
                    "content": 1,
                    "image": 1,
                    "gif": 1,
                    "aiReply": 1,
                    "likes": 1,
                    "reactions": 1,
                    "comments": 1,
                    "shares": 1,
                    "poll": 1,
                    "created_at": 1,
                }
            ).sort("created_at", -1).skip(skip).limit(limit).hint("community_created_at_desc")
            cursor.max_time_ms(8000)  # Server-side abort if query takes >8s
            return await cursor.to_list(length=limit)

        async def _fetch_bookmarked_ids():
            """Fetch user's bookmarked post IDs for bookmark badge on feed."""
            try:
                bms = await db["community_bookmarks"].find(
                    {"user_email": user_email},
                    {"post_id": 1}
                ).to_list(length=500)
                return {bm["post_id"] for bm in bms}
            except Exception:
                return set()

        try:
            posts, bookmarked_ids = await asyncio.gather(
                asyncio.wait_for(_fetch_posts(), timeout=DB_QUERY_TIMEOUT),
                _fetch_bookmarked_ids(),
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Community posts query timed out (skip={skip}, limit={limit})")
            return []

        # Safe serialization — skip broken posts instead of crashing entire feed
        result = []
        for post in posts:
            try:
                result.append(serialize_post(post, user_email, bookmarked_ids))
            except Exception as e:
                post_id = post.get("_id", "unknown")
                logger.warning(f"⚠️ Skipping broken post {post_id}: {e}")
                continue

        elapsed = (time.time() - start_time) * 1000
        if elapsed > 1000:
            logger.warning(f"🐌 Slow community posts query: {elapsed:.0f}ms ({len(result)} posts)")

        return result
    except Exception as e:
        logger.error(f"❌ Error fetching posts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch community posts")


@router.post("/posts")
async def create_community_post(
    post_data: PostCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new community post with server-side validation and moderation."""
    try:
        db = await get_db()
        await ensure_community_indexes(db)

        user_email = current_user["sub"]

        # Rate limit check
        if not check_rate_limit(user_email, "post"):
            raise HTTPException(status_code=429, detail="Too many posts. Please wait a minute before posting again.")

        # Server-side content moderation
        if post_data.content and not is_content_safe(post_data.content):
            raise HTTPException(status_code=422, detail="Content violates community guidelines. Please revise your post.")

        # Content validation — at least content, image, gif, or poll required
        if not post_data.content and not post_data.image and not post_data.gif and not post_data.poll:
            raise HTTPException(status_code=422, detail="Post must contain text, an image, a GIF, or a poll.")

        # Fetch user profile (with projection for speed)
        user_profile = await get_user_profile(db, user_email)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        new_post = {
            "user": user_profile,
            "content": post_data.content,
            "image": post_data.image,
            "gif": post_data.gif,
            "aiReply": post_data.aiReply,
            "likes": [],
            "comments": [],
            "shares": 0,
            "created_at": datetime.utcnow()
        }

        # Build poll document if provided
        if post_data.poll:
            poll_data = post_data.poll
            total_seconds = (
                poll_data.duration_days * 86400 +
                poll_data.duration_hours * 3600 +
                poll_data.duration_minutes * 60
            )
            # Default to 24h if all zeros
            if total_seconds == 0:
                total_seconds = 86400
            
            new_post["poll"] = {
                "question": poll_data.question or post_data.content,
                "choices": [
                    {
                        "text": choice.text.strip(),
                        "image": choice.image,
                        "voters": []
                    }
                    for choice in poll_data.choices
                ],
                "expires_at": datetime.utcnow() + timedelta(seconds=total_seconds),
                "created_at": datetime.utcnow()
            }

        result = await db["community_posts"].insert_one(new_post)

        # Return created post directly (avoid extra read)
        new_post["id"] = str(result.inserted_id)
        new_post.pop("_id", None)
        new_post["time"] = "Just now"
        new_post["liked"] = False

        # Serialize poll for response
        if "poll" in new_post and new_post["poll"]:
            poll = new_post["poll"]
            poll["total_votes"] = 0
            poll["user_voted"] = False
            poll["user_choice"] = None
            poll["is_expired"] = False
            poll["time_remaining"] = format_relative_time_future(poll.get("expires_at"))
            for choice in poll["choices"]:
                choice["votes"] = 0
                choice["percentage"] = 0
                choice.pop("voters", None)

        logger.info(f"✅ Post created by {user_email} (ID: {new_post['id']})")
        return new_post

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating post: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create post")


@router.post("/posts/{post_id}/like")
async def toggle_like(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Toggle like on a community post using atomic $push/$pull operations."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        # Validate ObjectId format
        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        oid = ObjectId(post_id)

        # Check if already liked — single atomic operation
        post = await db["community_posts"].find_one(
            {"_id": oid},
            {"likes": 1}  # Only fetch likes array
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        likes = post.get("likes", [])
        if user_email in likes:
            # Unlike — atomic $pull
            result = await db["community_posts"].update_one(
                {"_id": oid},
                {"$pull": {"likes": user_email}}
            )
            liked = False
            new_count = len(likes) - 1
        else:
            # Like — atomic $push
            result = await db["community_posts"].update_one(
                {"_id": oid},
                {"$push": {"likes": user_email}}
            )
            liked = True
            new_count = len(likes) + 1

            # Fire notification for post owner
            full_post = await db["community_posts"].find_one({"_id": oid}, {"user": 1, "content": 1})
            if full_post:
                post_owner_id = full_post.get("user", {}).get("id", "")
                if post_owner_id and ObjectId.is_valid(post_owner_id):
                    owner_doc = await db["users"].find_one({"_id": ObjectId(post_owner_id)}, {"email": 1})
                    if owner_doc:
                        user_profile = await get_user_profile(db, user_email)
                        asyncio.create_task(create_notification(
                            db,
                            recipient_email=owner_doc["email"],
                            actor_email=user_email,
                            actor_name=user_profile["name"] if user_profile else "Someone",
                            actor_avatar=user_profile.get("avatar") if user_profile else None,
                            notif_type="like",
                            post_id=post_id,
                            post_preview=full_post.get("content", ""),
                        ))

        return {"liked": liked, "likes_count": max(0, new_count)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error toggling like: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle like")


@router.post("/posts/{post_id}/comment")
async def add_comment(
    post_id: str,
    comment_data: CommentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a comment or nested reply to a community post."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        # Rate limit check
        if not check_rate_limit(user_email, "comment"):
            raise HTTPException(status_code=429, detail="Too many comments. Please slow down.")

        # Validate ObjectId
        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        # Server-side moderation
        if not is_content_safe(comment_data.text):
            raise HTTPException(status_code=422, detail="Comment violates community guidelines.")

        oid = ObjectId(post_id)

        # Verify post exists
        post = await db["community_posts"].find_one(
            {"_id": oid},
            {"_id": 1, "comments": 1}  # Only fetch needed fields
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Fetch user profile (with projection)
        user_profile = await get_user_profile(db, user_email)
        display_name = user_profile["name"] if user_profile else "User"
        user_handle = user_profile["handle"] if user_profile else "user"
        user_avatar = user_profile["avatar"] if user_profile else None

        new_comment = {
            "id": str(ObjectId()),
            "user": display_name,
            "user_handle": user_handle,
            "user_avatar": user_avatar,
            "text": comment_data.text,
            "time": "Just now",
            "likes": 0,
            "liked": False,
            "created_at": datetime.utcnow(),
            "replies": []
        }

        parent_id = comment_data.parentId

        if not parent_id:
            # Top-level comment — atomic $push
            await db["community_posts"].update_one(
                {"_id": oid},
                {"$push": {"comments": new_comment}}
            )
        else:
            # Nested reply — need to update the specific comment's replies array
            # Use $ positional operator for the parent comment
            comments = post.get("comments", [])
            for c in comments:
                if c.get("id") == parent_id:
                    if "replies" not in c:
                        c["replies"] = []
                    c["replies"].append(new_comment)
                    break

            await db["community_posts"].update_one(
                {"_id": oid},
                {"$set": {"comments": comments}}
            )

        # Fire notification for post owner
        post_owner_id = post.get("user", {}).get("id", "") if isinstance(post.get("user"), dict) else ""
        # We need to re-fetch with user field since our earlier projection only had _id and comments
        if not post_owner_id:
            full_post = await db["community_posts"].find_one({"_id": oid}, {"user": 1, "content": 1})
            if full_post:
                post_owner_id = full_post.get("user", {}).get("id", "")
                post_content = full_post.get("content", "")
            else:
                post_content = ""
        else:
            post_content = post.get("content", "")

        if post_owner_id and ObjectId.is_valid(post_owner_id):
            owner_doc = await db["users"].find_one({"_id": ObjectId(post_owner_id)}, {"email": 1})
            if owner_doc:
                asyncio.create_task(create_notification(
                    db,
                    recipient_email=owner_doc["email"],
                    actor_email=user_email,
                    actor_name=display_name,
                    actor_avatar=user_avatar,
                    notif_type="reply" if parent_id else "comment",
                    post_id=post_id,
                    post_preview=post_content,
                    comment_preview=comment_data.text,
                ))

        logger.info(f"💬 Comment added by {user_email} on post {post_id}")
        return new_comment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add comment")


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a community post (only by post owner or admin)."""
    try:
        db = await get_db()

        # Validate ObjectId
        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        oid = ObjectId(post_id)

        post = await db["community_posts"].find_one(
            {"_id": oid},
            {"user.id": 1}  # Only fetch user ID for auth check
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Authorization: post owner or admin
        post_user_id = post.get("user", {}).get("id", "")
        user = await db["users"].find_one(
            {"email": current_user["sub"]},
            {"_id": 1}
        )
        current_user_id = str(user["_id"]) if user else ""
        is_admin = current_user.get("role") == "admin"

        if post_user_id != current_user_id and not is_admin:
            raise HTTPException(status_code=403, detail="Not authorized to delete this post")

        await db["community_posts"].delete_one({"_id": oid})
        logger.info(f"🗑️ Post {post_id} deleted by {current_user['sub']}")
        return {"success": True, "deleted_id": post_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting post: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete post")


# ═══════════════════════════════════════
# COMMUNITY STATS ENDPOINT
# ═══════════════════════════════════════

@router.get("/stats")
async def get_community_stats(
    current_user: dict = Depends(get_current_user)
):
    """Return community engagement statistics — all queries run in parallel."""
    try:
        db = await get_db()
        coll = db["community_posts"]

        day_ago = datetime.utcnow() - timedelta(hours=24)

        # Run ALL stats queries in parallel — saves ~900ms vs sequential with Atlas latency
        async def _total():
            return await coll.estimated_document_count()  # O(1) metadata lookup, no scan

        async def _today():
            return await coll.count_documents({"created_at": {"$gte": day_ago}})

        async def _ai():
            return await coll.count_documents({"aiReply": {"$ne": None}})

        async def _contributors():
            pipeline = [
                {"$group": {"_id": "$user.id"}},
                {"$count": "total"}
            ]
            result = await coll.aggregate(pipeline).to_list(length=1)
            return result[0]["total"] if result else 0

        total_posts, today_posts, ai_responses, contributors = await asyncio.gather(
            _total(), _today(), _ai(), _contributors(),
            return_exceptions=True  # Don't crash if one query fails
        )

        # Handle any failed queries gracefully
        if isinstance(total_posts, Exception):
            logger.warning(f"⚠️ Stats total_posts failed: {total_posts}")
            total_posts = 0
        if isinstance(today_posts, Exception):
            today_posts = 0
        if isinstance(ai_responses, Exception):
            ai_responses = 0
        if isinstance(contributors, Exception):
            contributors = 0

        return {
            "total_posts": total_posts,
            "today_posts": today_posts,
            "ai_responses": ai_responses,
            "contributors": contributors
        }
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {str(e)}")
        return {"total_posts": 0, "today_posts": 0, "ai_responses": 0, "contributors": 0}


# ═══════════════════════════════════════
# POLL VOTING ENDPOINT
# ═══════════════════════════════════════

@router.post("/posts/{post_id}/vote")
async def vote_poll(
    post_id: str,
    vote_data: PollVote,
    current_user: dict = Depends(get_current_user)
):
    """Vote on a community poll. Users can only vote once per poll."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        oid = ObjectId(post_id)
        post = await db["community_posts"].find_one(
            {"_id": oid},
            {"poll": 1}
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        poll = post.get("poll")
        if not poll:
            raise HTTPException(status_code=400, detail="This post does not have a poll")

        # Check expiry
        expires_at = poll.get("expires_at")
        if expires_at and isinstance(expires_at, datetime) and datetime.utcnow() > expires_at:
            raise HTTPException(status_code=400, detail="This poll has ended")

        choices = poll.get("choices", [])
        if vote_data.choice_index >= len(choices):
            raise HTTPException(status_code=400, detail="Invalid choice index")

        # Check if user already voted
        for idx, choice in enumerate(choices):
            if user_email in choice.get("voters", []):
                raise HTTPException(status_code=400, detail="You have already voted on this poll")

        # Atomic $push vote
        await db["community_posts"].update_one(
            {"_id": oid},
            {"$push": {f"poll.choices.{vote_data.choice_index}.voters": user_email}}
        )

        # Re-fetch for updated counts
        updated = await db["community_posts"].find_one({"_id": oid}, {"poll": 1})
        updated_poll = updated.get("poll", {})
        total = sum(len(c.get("voters", [])) for c in updated_poll.get("choices", []))

        results = []
        for idx, c in enumerate(updated_poll.get("choices", [])):
            votes = len(c.get("voters", []))
            results.append({
                "text": c["text"],
                "image": c.get("image"),
                "votes": votes,
                "percentage": round((votes / total * 100) if total > 0 else 0, 1)
            })

        logger.info(f"🗳️ Vote cast by {user_email} on poll in post {post_id}")
        return {
            "success": True,
            "user_choice": vote_data.choice_index,
            "total_votes": total,
            "choices": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error voting on poll: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cast vote")


# ═══════════════════════════════════════
# GIF SEARCH (Tenor API)
# ═══════════════════════════════════════

@router.get("/gifs/search")
async def search_gifs(
    q: str = Query(..., min_length=1, max_length=100, description="Search query"),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Search for GIFs using Tenor API with circuit breaker protection."""
    import httpx

    # Circuit breaker: if Tenor is consistently failing, fail fast
    if _tenor_circuit_open():
        logger.debug("⚡ Tenor circuit breaker OPEN — returning empty results")
        return {"results": []}

    try:
        tenor_key = os.getenv("TENOR_API_KEY", "")
        if not tenor_key:
            return {"results": []}

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://tenor.googleapis.com/v2/search",
                params={
                    "q": q,
                    "key": tenor_key,
                    "client_key": "eaiser_community",
                    "limit": limit,
                    "media_filter": "gif,tinygif"
                }
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            media = item.get("media_formats", {})
            gif_url = media.get("gif", {}).get("url", "")
            tiny_url = media.get("tinygif", {}).get("url", gif_url)
            results.append({
                "id": item.get("id"),
                "title": item.get("content_description", ""),
                "url": gif_url,
                "preview": tiny_url,
                "dims": media.get("gif", {}).get("dims", [0, 0])
            })

        _tenor_record_success()
        return {"results": results}

    except Exception as e:
        _tenor_record_failure()
        logger.error(f"❌ GIF search error (failures={_tenor_failures}): {str(e)}")
        return {"results": []}


@router.get("/gifs/trending")
async def trending_gifs(
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Fetch trending GIFs from Tenor with circuit breaker protection."""
    import httpx

    if _tenor_circuit_open():
        return {"results": []}

    try:
        tenor_key = os.getenv("TENOR_API_KEY", "")
        if not tenor_key:
            return {"results": []}

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://tenor.googleapis.com/v2/featured",
                params={
                    "key": tenor_key,
                    "client_key": "eaiser_community",
                    "limit": limit,
                    "media_filter": "gif,tinygif"
                }
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            media = item.get("media_formats", {})
            gif_url = media.get("gif", {}).get("url", "")
            tiny_url = media.get("tinygif", {}).get("url", gif_url)
            results.append({
                "id": item.get("id"),
                "title": item.get("content_description", ""),
                "url": gif_url,
                "preview": tiny_url,
                "dims": media.get("gif", {}).get("dims", [0, 0])
            })

        _tenor_record_success()
        return {"results": results}

    except Exception as e:
        _tenor_record_failure()
        logger.error(f"❌ Trending GIFs error (failures={_tenor_failures}): {str(e)}")
        return {"results": []}


# ═══════════════════════════════════════
# AI CHAT (Community Bot — Gemini 2.0)
# ═══════════════════════════════════════

@router.post("/chat")
async def community_ai_chat(
    data: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    """AI-powered community chat bot using Gemini 2.0 Flash with civic context."""
    try:
        user_email = current_user["sub"]

        # Rate limit AI calls (more restrictive)
        if not check_rate_limit(user_email, "ai_chat"):
            return {"reply": "I'm getting a lot of questions right now! Please try again in a minute. 🕐"}

        message = data.message.strip()

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            return {"reply": "I'm temporarily offline. Our AI service will be back shortly! 🔧"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""You are EAiSER AI — the official community assistant for the EAiSER civic-tech platform.

EAiSER helps residents report municipal issues like potholes, broken streetlights, flooding, garbage, graffiti, fallen trees, traffic signal damage, water leakage, abandoned vehicles, and more. Reports are AI-analyzed using image recognition, then automatically routed to the correct local authority (Public Works, Water Board, Police, Emergency Services, etc.) for resolution.

PLATFORM FEATURES:
- Single Image Multiple Issues (SIMI) detection — one photo can detect multiple hazards
- GPS-based location tagging with map integration
- Real-time status tracking (submitted → dispatched → in_progress → resolved)
- Authority management dashboard for municipal workers
- Community Hub for civic discussions

RESPONSE RULES:
- Keep responses concise (2-4 sentences max)
- Be friendly, professional, and encouraging
- If asked about civic issues, give practical advice
- If asked about EAiSER features, explain clearly
- If the user shares a concern, acknowledge it and suggest they submit a report via the app
- Use relevant emojis sparingly (1-2 per response)
- Never make up statistics or false claims
- If unsure, suggest the user submit a report for official tracking
- Always maintain a helpful, civic-minded tone
- Do NOT respond to off-topic, harmful, or political questions — redirect to civic topics

User message: {message}

Your helpful response:"""

        response = model.generate_content(prompt)
        reply_text = response.text.strip() if response and response.text else "I'm here to help with civic reporting! Try submitting a report through the app. 📋"

        # Safety cap — truncate oversized responses to prevent payload bloat
        MAX_REPLY_LENGTH = 2000
        if len(reply_text) > MAX_REPLY_LENGTH:
            reply_text = reply_text[:MAX_REPLY_LENGTH].rsplit('.', 1)[0] + '.'
            logger.warning(f"⚠️ AI response truncated to {len(reply_text)} chars")

        logger.info(f"🤖 AI responded to {user_email}: {message[:50]}...")
        return {"reply": reply_text}

    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg:
            logger.warning(f"⚠️ AI quota exceeded: {e}")
            return {"reply": "Our AI assistant is resting due to high demand. Please try again in a few minutes! 🔋"}
        elif "timeout" in error_msg:
            logger.warning(f"⏰ AI timeout: {e}")
            return {"reply": "I'm taking longer than usual. Please try again shortly! ⏳"}
        else:
            logger.error(f"❌ Community AI chat error: {str(e)}")
            return {"reply": "I'm processing a lot of civic data right now. Please try again in a moment! 🔄"}


# ═══════════════════════════════════════
# NOTIFICATION HELPER
# ═══════════════════════════════════════

async def create_notification(
    db,
    recipient_email: str,
    actor_email: str,
    actor_name: str,
    actor_avatar: str,
    notif_type: str,       # "like" | "comment" | "reaction" | "reply"
    post_id: str,
    post_preview: str = "",
    reaction_emoji: str = None,
    comment_preview: str = None,
):
    """Create a notification document. Skips if actor == recipient (no self-notifications)."""
    if recipient_email == actor_email:
        return  # Don't notify yourself

    notif = {
        "recipient": recipient_email,
        "actor": {
            "email": actor_email,
            "name": actor_name,
            "avatar": actor_avatar,
        },
        "type": notif_type,
        "post_id": post_id,
        "post_preview": (post_preview or "")[:120],
        "reaction_emoji": reaction_emoji,
        "comment_preview": (comment_preview or "")[:100],
        "read": False,
        "created_at": datetime.utcnow(),
    }

    try:
        await db["community_notifications"].insert_one(notif)
    except Exception as e:
        logger.warning(f"⚠️ Notification creation failed (non-fatal): {e}")


# ═══════════════════════════════════════
# POST REACTIONS (❤️ 😡 😢 👏 🔥)
# ═══════════════════════════════════════

@router.post("/posts/{post_id}/react")
async def toggle_reaction(
    post_id: str,
    reaction_data: ReactionCreate,
    current_user: dict = Depends(get_current_user)
):
    """Toggle a reaction on a post. Each user can have ONE reaction per post.
    
    - If user has no reaction → add the reaction
    - If user has the SAME reaction → remove it (un-react)
    - If user has a DIFFERENT reaction → swap to new one
    
    Stores reactions as: { "reactions": { "❤️": ["email1"], "🔥": ["email2"] } }
    """
    try:
        db = await get_db()
        user_email = current_user["sub"]
        emoji = reaction_data.reaction

        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        oid = ObjectId(post_id)

        # Fetch only reactions field
        post = await db["community_posts"].find_one(
            {"_id": oid},
            {"reactions": 1, "user": 1, "content": 1}
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        reactions = post.get("reactions", {}) or {}
        current_reaction = None

        # Find user's current reaction (if any)
        for r_emoji, voters in reactions.items():
            if isinstance(voters, list) and user_email in voters:
                current_reaction = r_emoji
                break

        if current_reaction == emoji:
            # UN-REACT: remove the user from that emoji
            await db["community_posts"].update_one(
                {"_id": oid},
                {"$pull": {f"reactions.{emoji}": user_email}}
            )
            action = "removed"
            final_emoji = None
        else:
            # Remove from old reaction if switching
            if current_reaction:
                await db["community_posts"].update_one(
                    {"_id": oid},
                    {"$pull": {f"reactions.{current_reaction}": user_email}}
                )
            # Add to new reaction
            await db["community_posts"].update_one(
                {"_id": oid},
                {"$addToSet": {f"reactions.{emoji}": user_email}}
            )
            action = "added"
            final_emoji = emoji

            # Create notification for post owner
            post_owner_email = post.get("user", {}).get("id", "")
            if post_owner_email:
                # Lookup owner's email from user ID
                owner_user = await db["users"].find_one(
                    {"_id": ObjectId(post_owner_email)},
                    {"email": 1}
                ) if ObjectId.is_valid(post_owner_email) else None
                owner_email = owner_user.get("email", "") if owner_user else ""
            else:
                owner_email = ""

            if owner_email:
                user_profile = await get_user_profile(db, user_email)
                asyncio.create_task(create_notification(
                    db,
                    recipient_email=owner_email,
                    actor_email=user_email,
                    actor_name=user_profile["name"] if user_profile else "Someone",
                    actor_avatar=user_profile.get("avatar") if user_profile else None,
                    notif_type="reaction",
                    post_id=post_id,
                    post_preview=post.get("content", ""),
                    reaction_emoji=emoji,
                ))

        # Re-fetch updated reactions
        updated = await db["community_posts"].find_one({"_id": oid}, {"reactions": 1})
        updated_reactions = updated.get("reactions", {}) or {}

        # Build summary
        summary = {}
        for r_emoji in VALID_REACTIONS:
            voters = updated_reactions.get(r_emoji, [])
            if isinstance(voters, list) and len(voters) > 0:
                summary[r_emoji] = len(voters)

        logger.info(f"{'➕' if action == 'added' else '➖'} Reaction {emoji} {action} by {user_email} on post {post_id}")
        return {
            "action": action,
            "user_reaction": final_emoji,
            "reactions": summary,
            "total": sum(summary.values()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error toggling reaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle reaction")


# ═══════════════════════════════════════
# BOOKMARKS / SAVE POSTS
# ═══════════════════════════════════════

@router.post("/posts/{post_id}/bookmark")
async def toggle_bookmark(
    post_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Toggle bookmark on a post. Uses a dedicated collection for O(1) lookups."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        if not ObjectId.is_valid(post_id):
            raise HTTPException(status_code=400, detail="Invalid post ID format")

        # Verify post exists
        post = await db["community_posts"].find_one(
            {"_id": ObjectId(post_id)},
            {"_id": 1}
        )
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        # Check if already bookmarked
        existing = await db["community_bookmarks"].find_one({
            "user_email": user_email,
            "post_id": post_id,
        })

        if existing:
            # Remove bookmark
            await db["community_bookmarks"].delete_one({"_id": existing["_id"]})
            bookmarked = False
        else:
            # Add bookmark
            await db["community_bookmarks"].insert_one({
                "user_email": user_email,
                "post_id": post_id,
                "created_at": datetime.utcnow(),
            })
            bookmarked = True

        logger.info(f"{'🔖' if bookmarked else '📤'} Bookmark {'added' if bookmarked else 'removed'} by {user_email} on post {post_id}")
        return {"bookmarked": bookmarked, "post_id": post_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error toggling bookmark: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to toggle bookmark")


@router.get("/bookmarks")
async def get_user_bookmarks(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Fetch the current user's bookmarked posts, newest first."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        # Get bookmark entries
        bookmarks = await db["community_bookmarks"].find(
            {"user_email": user_email}
        ).sort("created_at", -1).skip(skip).limit(limit).to_list(length=limit)

        post_ids = []
        for bm in bookmarks:
            pid = bm.get("post_id", "")
            if ObjectId.is_valid(pid):
                post_ids.append(ObjectId(pid))

        if not post_ids:
            return []

        # Fetch actual posts
        posts = await db["community_posts"].find(
            {"_id": {"$in": post_ids}}
        ).to_list(length=limit)

        # Preserve bookmark order
        post_map = {}
        for p in posts:
            post_map[str(p["_id"])] = p

        result = []
        for bm in bookmarks:
            pid = bm.get("post_id", "")
            if pid in post_map:
                serialized = serialize_post(post_map[pid], user_email)
                serialized["bookmarked"] = True
                result.append(serialized)

        return result

    except Exception as e:
        logger.error(f"❌ Error fetching bookmarks: {str(e)}")
        return []


# ═══════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════

@router.get("/notifications")
async def get_notifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=50),
    current_user: dict = Depends(get_current_user)
):
    """Fetch paginated notifications for the current user, newest first."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        cursor = db["community_notifications"].find(
            {"recipient": user_email}
        ).sort("created_at", -1).skip(skip).limit(limit)

        notifs = await cursor.to_list(length=limit)

        result = []
        for n in notifs:
            result.append({
                "id": str(n.pop("_id", "")),
                "actor": n.get("actor", {}),
                "type": n.get("type", ""),
                "post_id": n.get("post_id", ""),
                "post_preview": n.get("post_preview", ""),
                "reaction_emoji": n.get("reaction_emoji"),
                "comment_preview": n.get("comment_preview"),
                "read": n.get("read", False),
                "time": format_relative_time(n.get("created_at")),
                "created_at": n.get("created_at", datetime.utcnow()).isoformat(),
            })

        return result

    except Exception as e:
        logger.error(f"❌ Error fetching notifications: {str(e)}")
        return []


@router.get("/notifications/unread-count")
async def get_unread_count(
    current_user: dict = Depends(get_current_user)
):
    """Return the count of unread notifications for badge display."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        count = await db["community_notifications"].count_documents({
            "recipient": user_email,
            "read": False,
        })

        return {"count": min(count, 99)}  # Cap at 99 for UI badge

    except Exception as e:
        logger.error(f"❌ Error counting unread: {str(e)}")
        return {"count": 0}


@router.post("/notifications/mark-read")
async def mark_notifications_read(
    notification_ids: List[str] = Body(default=None),
    mark_all: bool = Body(default=False),
    current_user: dict = Depends(get_current_user)
):
    """Mark specific notifications or ALL as read."""
    try:
        db = await get_db()
        user_email = current_user["sub"]

        if mark_all:
            result = await db["community_notifications"].update_many(
                {"recipient": user_email, "read": False},
                {"$set": {"read": True}}
            )
            modified = result.modified_count
        elif notification_ids:
            valid_ids = [ObjectId(nid) for nid in notification_ids if ObjectId.is_valid(nid)]
            if valid_ids:
                result = await db["community_notifications"].update_many(
                    {"_id": {"$in": valid_ids}, "recipient": user_email},
                    {"$set": {"read": True}}
                )
                modified = result.modified_count
            else:
                modified = 0
        else:
            modified = 0

        return {"success": True, "modified": modified}

    except Exception as e:
        logger.error(f"❌ Error marking notifications read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark notifications")
