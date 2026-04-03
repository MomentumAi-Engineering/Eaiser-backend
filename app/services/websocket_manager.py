from typing import List, Dict, Any
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Store active connections: {user_email: [WebSocket, ...]}
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_email: str):
        await websocket.accept()
        if user_email not in self.active_connections:
            self.active_connections[user_email] = []
        self.active_connections[user_email].append(websocket)
        logger.info(f"🔌 WebSocket connected for user: {user_email}")

    def disconnect(self, websocket: WebSocket, user_email: str):
        if user_email in self.active_connections:
            try:
                self.active_connections[user_email].remove(websocket)
                if not self.active_connections[user_email]:
                    del self.active_connections[user_email]
                logger.info(f"🔌 WebSocket disconnected for user: {user_email}")
            except ValueError:
                pass

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_user(self, user_email: str, message: Dict[str, Any]):
        """Send a real-time update to all active devices of a specific user."""
        if user_email in self.active_connections:
            payload = json.dumps(message)
            for connection in self.active_connections[user_email]:
                try:
                    await connection.send_text(payload)
                except Exception as e:
                    logger.warning(f"Failed to send WS message to {user_email}: {e}")

    async def handle_message(self, websocket: WebSocket, data: str):
        """Handle incoming messages like PINGs."""
        try:
            msg = json.loads(data)
            if msg.get("type") == "PING":
                await websocket.send_text(json.dumps({"type": "PONG"}))
        except Exception:
            pass

manager = ConnectionManager()
