"""
JarvisAI Caching Layer
Simple in-memory and Redis caching for LLM responses
"""

import hashlib
import json
import time
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with expiration"""

    def __init__(self, value: Any, ttl: int = 3600):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class MemoryCache:
    """In-memory cache implementation"""

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                return entry.value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        self._cache[key] = CacheEntry(value, ttl)

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class LLMCache:
    """
    Cache for LLM responses to reduce API costs
    """

    def __init__(
        self, use_redis: bool = False, redis_url: str = "redis://localhost:6379/0"
    ):
        self.memory_cache = MemoryCache()
        self.use_redis = use_redis
        self._redis = None

        if use_redis:
            try:
                import redis

                self._redis = redis.from_url(redis_url)
                logger.info("Redis cache initialized")
            except ImportError:
                logger.warning("redis package not installed, using memory cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")

    def _make_key(self, messages: list, model: str, temperature: float) -> str:
        """Generate cache key from request parameters"""
        key_data = {"messages": messages, "model": model, "temperature": temperature}
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm:{hashlib.sha256(key_str.encode()).hexdigest()[:32]}"

    def get(
        self, messages: list, model: str, temperature: float = 0.7
    ) -> Optional[str]:
        """Get cached response"""
        key = self._make_key(messages, model, temperature)

        if self._redis:
            try:
                cached = self._redis.get(key)
                if cached:
                    logger.debug(f"Cache hit (Redis): {key}")
                    return cached.decode() if isinstance(cached, bytes) else cached
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        result = self.memory_cache.get(key)
        if result:
            logger.debug(f"Cache hit (Memory): {key}")
        return result

    def set(
        self,
        messages: list,
        model: str,
        response: str,
        temperature: float = 0.7,
        ttl: int = 3600,
    ) -> None:
        """Cache response"""
        key = self._make_key(messages, model, temperature)

        if self._redis:
            try:
                self._redis.setex(key, ttl, response)
                logger.debug(f"Cached (Redis): {key}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        self.memory_cache.set(key, response, ttl)
        logger.debug(f"Cached (Memory): {key}")

    def invalidate(self, messages: list, model: str, temperature: float = 0.7) -> None:
        """Invalidate cached response"""
        key = self._make_key(messages, model, temperature)

        if self._redis:
            try:
                self._redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")

        self.memory_cache.delete(key)

    def clear(self) -> None:
        """Clear all cached responses"""
        if self._redis:
            try:
                keys = self._redis.keys("llm:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")

        self.memory_cache.clear()
        logger.info("Cache cleared")


llm_cache = LLMCache()
