import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from pymongo import monitoring

logger = logging.getLogger("performance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_command_timings = {}

class CommandLogger(monitoring.CommandListener):
    def started(self, event):
        _command_timings[event.request_id] = time.time()

    def succeeded(self, event):
        start_time = _command_timings.pop(event.request_id, None)
        if start_time:
            duration = (time.time() - start_time) * 1000
            logger.info(f"✅ MongoDB Query: {event.command_name} took {duration:.2f} ms")

    def failed(self, event):
        start_time = _command_timings.pop(event.request_id, None)
        if start_time:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ MongoDB Query Failed: {event.command_name} after {duration:.2f} ms")
        else:
            logger.error(f"❌ MongoDB Query Failed: {event.command_name}")

# Register listener
monitoring.register(CommandLogger())

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        logger.info(f"⏱️ Request {request.method} {request.url.path} took {process_time:.2f} ms")

        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
        return response
