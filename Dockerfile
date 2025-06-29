# Dockerfile for Jarvis AI Platform

# Use official Python slim image
FROM python:3.10-slim

# Install system dependencies (curl for healthcheck, build-essential for some Python packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up environment
RUN useradd -m -d /home/jarvisuser jarvisuser
ENV HOME=/home/jarvisuser \
    PATH="/home/jarvisuser/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory and permissions
WORKDIR /home/jarvisuser/app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Change ownership to non-root user
RUN chown -R jarvisuser:jarvisuser /home/jarvisuser

EXPOSE 8000

# Install gunicorn for production WSGI serving (as root, before switching user)
RUN pip install --no-cache-dir gunicorn && pip cache purge

# Switch to non-root user
USER jarvisuser

# Optional: define a volume for persistent data (logs, uploads, etc.)
# VOLUME ["/home/jarvisuser/app/data"]


# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

LABEL maintainer="Jarvis Maintainers <maintainers@jarvis.ai>"
LABEL org.opencontainers.image.source="https://github.com/jarvis-ai/jarvis"
LABEL org.opencontainers.image.description="Jarvis AI Platform - Robust, scalable, and production-ready AI orchestration."
LABEL org.opencontainers.image.version="1.0.0"

# Default command: use gunicorn for production (assumes Flask/FastAPI app as 'app' in api.py)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "api:app"]
