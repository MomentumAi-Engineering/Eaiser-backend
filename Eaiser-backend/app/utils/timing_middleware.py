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
            
            # Only log slow MongoDB queries to reduce noise
            if duration > 100:  # Log queries taking more than 100ms
                logger.warning(f"🐌 Slow MongoDB {event.command_name}: {duration:.2f} ms")
            elif duration > 50:  # Log moderately slow queries at info level
                logger.info(f"⚡ MongoDB {event.command_name}: {duration:.2f} ms")
            # Skip logging fast queries (< 50ms) to reduce noise

    def failed(self, event):
        start_time = _command_timings.pop(event.request_id, None)
        if start_time:
            duration = (time.time() - start_time) * 1000
            logger.error(f"❌ MongoDB {event.command_name} failed after {duration:.2f} ms")
        else:
            logger.error(f"❌ MongoDB {event.command_name} failed (duration unknown)")

# Register listener
monitoring.register(CommandLogger())

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        # Filter out noisy health check and static file requests
        path = request.url.path
        should_log = not any([
            path in ["/", "/health", "/favicon.ico", "/robots.txt"],
            path.startswith("/static/"),
            path.startswith("/assets/"),
            path.endswith(".ico"),
            path.endswith(".png"),
            path.endswith(".jpg"),
            path.endswith(".css"),
            path.endswith(".js")
        ])
        
        if should_log:
            if process_time > 1000:  # Log slow requests as warnings
                logger.warning(f"🐌 Slow request {request.method} {path} took {process_time:.2f} ms")
            else:
                logger.info(f"⏱️ Request {request.method} {path} took {process_time:.2f} ms")

        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
        return response
