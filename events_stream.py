"""
Real-time Server-Sent Events (SSE) stream for JarvisAI.

Provides live event streaming for:
- Training progress updates
- Autonomous mission step notifications
- Watcher alerts
- System health changes
- Custom event broadcasting
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.infra.auth_helpers import get_current_user
from src.infra.models_user import User


router = APIRouter(prefix="/events", tags=["Real-time Events"])


# --- Event Store ---

class EventStore:
    """In-memory event store with channel-based pub/sub."""

    def __init__(self):
        self._channels: Dict[str, List[Dict]] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._history_limit = 100

    def publish(self, channel: str, event_type: str, data: Any = None):
        """Publish an event to a channel."""
        event = {
            "id": int(time.time() * 1000),
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Store in history
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(event)
        if len(self._channels[channel]) > self._history_limit:
            self._channels[channel] = self._channels[channel][-self._history_limit:]

        # Notify subscribers
        if channel in self._subscribers:
            for queue in self._subscribers[channel]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def get_history(self, channel: str, limit: int = 50) -> List[Dict]:
        """Get recent events from a channel."""
        events = self._channels.get(channel, [])
        return events[-limit:]

    def subscribe(self, channel: str) -> asyncio.Queue:
        """Subscribe to a channel. Returns a queue that receives events."""
        queue = asyncio.Queue(maxsize=100)
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue):
        """Unsubscribe from a channel."""
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(queue)
            except ValueError:
                pass


# Global event store
event_store = EventStore()


# --- Pydantic Schemas ---

class EventPublishRequest(BaseModel):
    channel: str = "general"
    event_type: str = "message"
    data: Any = None


# --- Endpoints ---

@router.get("/stream/{channel}")
async def stream_events(
    channel: str,
    request: Request,
    last_id: Optional[int] = None,
):
    """Stream events from a channel using Server-Sent Events (SSE)."""
    queue = event_store.subscribe(channel)

    async def event_generator():
        # Send historical events if last_id specified
        if last_id:
            history = event_store.get_history(channel)
            for event in history:
                if event["id"] > last_id:
                    yield f"data: {json.dumps(event)}\n\n"

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            event_store.unsubscribe(channel, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/publish")
def publish_event(
    body: EventPublishRequest,
    current_user: User = Depends(get_current_user),
):
    """Publish an event to a channel."""
    event_store.publish(body.channel, body.event_type, body.data)
    return {"message": "Event published", "channel": body.channel}


@router.get("/history/{channel}")
def get_event_history(
    channel: str,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """Get recent events from a channel."""
    return {
        "channel": channel,
        "events": event_store.get_history(channel, limit),
    }


@router.get("/channels")
def list_channels(current_user: User = Depends(get_current_user)):
    """List all active channels."""
    return {
        "channels": list(event_store._channels.keys()),
        "subscriber_counts": {
            ch: len(subs) for ch, subs in event_store._subscribers.items()
        },
    }
