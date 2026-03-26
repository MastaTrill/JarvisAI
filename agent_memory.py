from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _get_redis_url() -> str:
    return os.getenv("REDIS_URL", "").strip()


@dataclass
class StoredMessage:
    role: str
    text: str

    def to_responses_input(self) -> Dict[str, Any]:
        return {"role": self.role, "content": [{"type": "input_text", "text": self.text}]}


class AgentMemory:
    """
    Redis-backed (preferred) memory for chat sessions.

    Stores a JSON list of messages per session_id under `jarvis:agent:<session_id>`.
    """

    def __init__(self, ttl_seconds: int = 24 * 3600) -> None:
        self._ttl_seconds = ttl_seconds
        self._redis = None
        redis_url = _get_redis_url()
        if redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self._redis = None

        # Fallback in-memory store for environments with no Redis.
        self._local: Dict[str, List[StoredMessage]] = {}

    def _key(self, session_id: str) -> str:
        return f"jarvis:agent:{session_id}"

    def load(self, session_id: str, max_messages: int = 12) -> List[StoredMessage]:
        if self._redis is None:
            return list(self._local.get(session_id, [])[-max_messages:])

        raw = self._redis.get(self._key(session_id))
        if not raw:
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        messages: List[StoredMessage] = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                text = item.get("text")
                if isinstance(role, str) and isinstance(text, str):
                    messages.append(StoredMessage(role=role, text=text))
        return messages[-max_messages:]

    def append(self, session_id: str, message: StoredMessage) -> None:
        if self._redis is None:
            self._local.setdefault(session_id, []).append(message)
            return

        key = self._key(session_id)
        existing = self.load(session_id, max_messages=200)
        existing.append(message)
        self._redis.set(key, json.dumps([m.__dict__ for m in existing]), ex=self._ttl_seconds)

