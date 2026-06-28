"""
Redis configuration for Jarvis AI (used for rate limiting, caching, and audit logging)
"""
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)
