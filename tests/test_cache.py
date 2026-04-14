"""
Tests for the LLM caching layer
"""

import pytest
from cache import MemoryCache, LLMCache, llm_cache


class TestMemoryCache:
    """Test in-memory cache"""

    def test_set_and_get(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_expiration(self):
        cache = MemoryCache()
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        import time

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_clear(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0

    def test_size(self):
        cache = MemoryCache()
        assert cache.size() == 0
        cache.set("key1", "value1")
        assert cache.size() == 1


class TestLLMCache:
    """Test LLM cache"""

    def test_cache_key_generation(self):
        messages = [{"role": "user", "content": "Hello"}]
        cache = LLMCache(use_redis=False)

        key1 = cache._make_key(messages, "gpt-4", 0.7)
        key2 = cache._make_key(messages, "gpt-4", 0.7)

        assert key1 == key2

    def test_different_messages_different_keys(self):
        cache = LLMCache(use_redis=False)

        key1 = cache._make_key([{"role": "user", "content": "Hello"}], "gpt-4", 0.7)
        key2 = cache._make_key([{"role": "user", "content": "World"}], "gpt-4", 0.7)

        assert key1 != key2

    def test_set_and_get(self):
        cache = LLMCache(use_redis=False)
        messages = [{"role": "user", "content": "test"}]

        cache.set(messages, "gpt-4", "Hello World", temperature=0.7)
        result = cache.get(messages, "gpt-4", 0.7)

        assert result == "Hello World"

    def test_cache_miss(self):
        cache = LLMCache(use_redis=False)
        messages = [{"role": "user", "content": "never seen this"}]

        result = cache.get(messages, "gpt-4", 0.7)

        assert result is None

    def test_invalidate(self):
        cache = LLMCache(use_redis=False)
        messages = [{"role": "user", "content": "test"}]

        cache.set(messages, "gpt-4", "Hello", temperature=0.7)
        cache.invalidate(messages, "gpt-4", 0.7)

        result = cache.get(messages, "gpt-4", 0.7)
        assert result is None

    def test_clear(self):
        cache = LLMCache(use_redis=False)

        cache.set([{"role": "user", "content": "test1"}], "gpt-4", "Hello1")
        cache.set([{"role": "user", "content": "test2"}], "gpt-4", "Hello2")

        cache.clear()

        assert cache.memory_cache.size() == 0


class TestLLMCacheIntegration:
    """Integration tests for LLM cache"""

    def test_global_cache_exists(self):
        assert llm_cache is not None
        assert hasattr(llm_cache, "get")
        assert hasattr(llm_cache, "set")
        assert hasattr(llm_cache, "clear")

    def test_cache_functions(self):
        test_messages = [{"role": "user", "content": "integration test"}]

        result = llm_cache.get(test_messages, "gpt-4", 0.7)

        llm_cache.set(test_messages, "gpt-4", "cached response", 0.7, 3600)

        result_after = llm_cache.get(test_messages, "gpt-4", 0.7)
        assert result_after == "cached response"

        llm_cache.clear()
