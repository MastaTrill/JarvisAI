"""
JarvisAI Celery Task Queue
Celery app and example background tasks
"""

from celery import Celery
import time
import os

# Redis broker URL (default: localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "jarvisai",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
)

if __name__ == "__main__":
    print(f"🚀 Celery app configured with broker: {REDIS_URL}")
    print("To start a worker: celery -A celery_app worker --loglevel=info")
