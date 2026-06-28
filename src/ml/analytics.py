"""
API Usage Analytics for JarvisAI.

Tracks and reports:
- Request counts per endpoint
- Response time percentiles
- Error rates
- Active users
- Token usage estimates
- Hourly/daily usage patterns
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from src.infra.auth_helpers import admin_required
from src.infra.models_user import User


router = APIRouter(prefix="/analytics", tags=["Analytics"])

# In-memory analytics store
_analytics = {
    "requests": [],  # List of {endpoint, method, status, duration_ms, timestamp, user}
    "start_time": datetime.now(timezone.utc),
}


def record_request(endpoint: str, method: str, status: int, duration_ms: float, user: str = "anonymous"):
    """Record an API request for analytics."""
    _analytics["requests"].append({
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
    })
    # Keep only last 10000 requests
    if len(_analytics["requests"]) > 10000:
        _analytics["requests"] = _analytics["requests"][-10000:]


@router.get("/overview")
def analytics_overview(
    current_user: User = Depends(admin_required),
):
    """Get API usage overview."""
    requests = _analytics["requests"]
    now = datetime.now(timezone.utc)
    uptime = (now - _analytics["start_time"]).total_seconds()

    # Last 24 hours
    day_ago = now - timedelta(hours=24)
    recent = [r for r in requests if datetime.fromisoformat(r["timestamp"]) > day_ago]

    # Endpoint stats
    endpoint_stats: Dict[str, Dict] = {}
    for r in requests:
        key = f"{r['method']} {r['endpoint']}"
        if key not in endpoint_stats:
            endpoint_stats[key] = {"count": 0, "errors": 0, "total_ms": 0}
        endpoint_stats[key]["count"] += 1
        if r["status"] >= 400:
            endpoint_stats[key]["errors"] += 1
        endpoint_stats[key]["total_ms"] += r["duration_ms"]

    # Calculate averages
    for key, stats in endpoint_stats.items():
        stats["avg_ms"] = round(stats["total_ms"] / stats["count"], 2) if stats["count"] > 0 else 0
        stats["error_rate"] = round(stats["errors"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0

    # Top endpoints by count
    top_endpoints = sorted(endpoint_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:20]

    # Unique users
    unique_users = set(r["user"] for r in requests)

    return {
        "total_requests": len(requests),
        "requests_last_24h": len(recent),
        "unique_users": len(unique_users),
        "uptime_seconds": round(uptime),
        "top_endpoints": [
            {"endpoint": k, **v} for k, v in top_endpoints
        ],
    }


@router.get("/endpoints")
def endpoint_analytics(
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(admin_required),
):
    """Get per-endpoint analytics."""
    requests = _analytics["requests"]

    endpoint_stats: Dict[str, Dict] = {}
    for r in requests:
        key = f"{r['method']} {r['endpoint']}"
        if key not in endpoint_stats:
            endpoint_stats[key] = {
                "count": 0, "errors": 0, "total_ms": 0,
                "min_ms": float("inf"), "max_ms": 0, "statuses": {},
            }
        s = endpoint_stats[key]
        s["count"] += 1
        if r["status"] >= 400:
            s["errors"] += 1
        s["total_ms"] += r["duration_ms"]
        s["min_ms"] = min(s["min_ms"], r["duration_ms"])
        s["max_ms"] = max(s["max_ms"], r["duration_ms"])
        s["statuses"][str(r["status"])] = s["statuses"].get(str(r["status"]), 0) + 1

    results = []
    for key, s in endpoint_stats.items():
        results.append({
            "endpoint": key,
            "count": s["count"],
            "avg_ms": round(s["total_ms"] / s["count"], 2) if s["count"] > 0 else 0,
            "min_ms": round(s["min_ms"], 2) if s["min_ms"] != float("inf") else 0,
            "max_ms": round(s["max_ms"], 2),
            "error_rate": round(s["errors"] / s["count"] * 100, 2) if s["count"] > 0 else 0,
            "statuses": s["statuses"],
        })

    results.sort(key=lambda x: x["count"], reverse=True)
    return {"endpoints": results[:limit]}


@router.get("/timeline")
def timeline_analytics(
    hours: int = Query(24, ge=1, le=168),
    current_user: User = Depends(admin_required),
):
    """Get request timeline (requests per hour)."""
    requests = _analytics["requests"]
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    # Group by hour
    hourly: Dict[str, int] = {}
    for r in requests:
        ts = datetime.fromisoformat(r["timestamp"])
        if ts > cutoff:
            hour_key = ts.strftime("%Y-%m-%d %H:00")
            hourly[hour_key] = hourly.get(hour_key, 0) + 1

    # Fill in missing hours
    timeline = []
    for i in range(hours):
        hour = (now - timedelta(hours=hours - i - 1)).strftime("%Y-%m-%d %H:00")
        timeline.append({"hour": hour, "requests": hourly.get(hour, 0)})

    return {"timeline": timeline, "total": sum(h["requests"] for h in timeline)}
