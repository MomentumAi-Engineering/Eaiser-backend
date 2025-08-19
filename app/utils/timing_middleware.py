import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from pymongo import monitoring

logger = logging.getLogger("performance")

class CommandLogger(monitoring.CommandListener):
    def started(self, event):
        event.command["__start_time__"] = time.time()

    def succeeded(self, event):
        start_time = event.command.get("__start_time__")
        if start_time:
            duration = (time.time() - start_time) * 1000
            logger.info(f"✅ MongoDB Query: {event.command_name} took {duration:.2f} ms")

    def failed(self, event):
        logger.error(f"❌ MongoDB Query Failed: {event.command_name}")

monitoring.register(CommandLogger())

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"⏱️ Request {request.method} {request.url.path} took {process_time:.2f} ms")
        return response
