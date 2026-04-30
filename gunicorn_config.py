"""
🚀 Gunicorn Production Configuration for EAiSER Backend
-------------------------------------------------------
Usage: gunicorn -c gunicorn_config.py app.main:app

For Windows (Gunicorn not supported natively):
  Use: uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8005
  Or:  python app/main.py  (auto-detects CPU cores)
"""

import multiprocessing
import os

# ==============================
# Server Socket
# ==============================
bind = f"0.0.0.0:{os.environ.get('PORT', '8005')}"
backlog = 2048  # Pending connections queue

# ==============================
# Workers (KEY for 10K users)
# ==============================
# Formula: (2 × CPU cores) + 1 for I/O-bound apps
cpu_cores = multiprocessing.cpu_count()
workers = int(os.environ.get("WEB_CONCURRENCY", (2 * cpu_cores) + 1))
worker_class = "uvicorn.workers.UvicornWorker"  # Async workers

# Each worker handles ~500 concurrent connections
# 4 workers = ~2000 concurrent, 8 workers = ~4000 concurrent
# For 10K: use 8-12 workers on a 4-core machine

# ==============================
# Worker Connections
# ==============================
worker_connections = 1000  # Max simultaneous connections per worker
max_requests = 5000       # Recycle worker after N requests (prevents memory leaks)
max_requests_jitter = 500 # Random jitter to prevent all workers restarting at once

# ==============================
# Timeouts
# ==============================
timeout = 120        # Worker timeout (AI calls can take up to 30s)
graceful_timeout = 30  # Time to finish requests before force kill
keepalive = 5        # Keep-alive connections

# ==============================
# Logging
# ==============================
loglevel = "info"
accesslog = "-"      # stdout
errorlog = "-"       # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ==============================
# Performance
# ==============================
preload_app = True   # Load app before forking workers (saves memory)
forwarded_allow_ips = "*"  # Trust proxy headers
proxy_protocol = False
