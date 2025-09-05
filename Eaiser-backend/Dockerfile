# 🚀 SnapFix FastAPI Dockerfile for Scalable Deployment
# Optimized for 1 lakh concurrent users with multi-stage build

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r snapfix && useradd -r -g snapfix snapfix

# Create app directory
WORKDIR /app

# Copy application code
COPY app/ /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/static /app/uploads && \
    chown -R snapfix:snapfix /app

# Switch to non-root user
USER snapfix

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Expose port
EXPOSE 10000

# Production-optimized startup command
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "10000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--timeout-keep-alive", "5", \
     "--access-log", \
     "--log-level", "info"]

# Alternative: Gunicorn with Uvicorn workers (for better performance)
# CMD ["gunicorn", "main:app", \
#      "-w", "4", \
#      "-k", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:10000", \
#      "--max-requests", "1000", \
#      "--max-requests-jitter", "100", \
#      "--timeout", "120", \
#      "--keep-alive", "5", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-", \
#      "--log-level", "info"]