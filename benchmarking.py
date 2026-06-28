"""
Performance Benchmarking System for Jarvis AI.

Provides:
- API endpoint latency benchmarking
- Model inference speed testing
- System resource monitoring
- Benchmark history and comparison
"""

import asyncio
import time
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON

from src.infra.db_config import Base as ConfigBase
from src.infra.database import get_db
from src.infra.auth_helpers import admin_required, get_current_user
from src.infra.models_user import User


# --- Database Model ---

class BenchmarkRun(ConfigBase):
    __tablename__ = "benchmark_runs"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    benchmark_type = Column(String(50))  # api_latency, model_inference, system
    target = Column(String(500))  # endpoint path or model name
    iterations = Column(Integer, default=100)

    # Results
    avg_latency_ms = Column(Float)
    min_latency_ms = Column(Float)
    max_latency_ms = Column(Float)
    median_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    std_dev_ms = Column(Float)
    requests_per_second = Column(Float)
    error_rate = Column(Float, default=0.0)

    # System state during benchmark
    system_snapshot = Column(JSON)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))


# --- Pydantic Schemas ---

class BenchmarkRequest(BaseModel):
    name: str = "default"
    iterations: int = 100
    warmup_iterations: int = 10


class BenchmarkResult(BaseModel):
    id: str
    name: str
    benchmark_type: str
    target: str
    iterations: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_rate: float
    created_at: str


# --- Router ---

router = APIRouter(prefix="/benchmarks", tags=["Performance Benchmarking"])


@router.post("/api")
async def benchmark_api_endpoint(
    endpoint: str = Query(..., description="API endpoint path to benchmark, e.g. /health"),
    method: str = Query("GET", description="HTTP method"),
    iterations: int = Query(100, ge=1, le=10000),
    warmup: int = Query(10, ge=0, le=1000),
    current_user: User = Depends(admin_required),
):
    """Benchmark an API endpoint's latency."""
    import uuid
    import psutil

    from main_api import app
    from starlette.testclient import TestClient

    client = TestClient(app)
    latencies = []
    errors = 0

    # Warmup
    for _ in range(warmup):
        try:
            if method.upper() == "GET":
                client.get(endpoint)
            elif method.upper() == "POST":
                client.post(endpoint)
        except Exception:
            pass

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            if method.upper() == "GET":
                resp = client.get(endpoint)
            elif method.upper() == "POST":
                resp = client.post(endpoint)
            else:
                raise HTTPException(400, f"Unsupported method: {method}")
            if resp.status_code >= 500:
                errors += 1
        except Exception:
            errors += 1
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    if not latencies:
        raise HTTPException(500, "All benchmark requests failed")

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    avg = statistics.mean(sorted_lat)
    error_rate = errors / iterations

    result = {
        "id": str(uuid.uuid4())[:8],
        "endpoint": endpoint,
        "method": method.upper(),
        "iterations": iterations,
        "avg_latency_ms": round(avg, 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "median_latency_ms": round(statistics.median(sorted_lat), 3),
        "p95_latency_ms": round(sorted_lat[int(n * 0.95)], 3),
        "p99_latency_ms": round(sorted_lat[int(n * 0.99)], 3),
        "std_dev_ms": round(statistics.stdev(sorted_lat), 3) if n > 1 else 0,
        "requests_per_second": round(1000 / avg * (1 - error_rate), 1) if avg > 0 else 0,
        "error_rate": round(error_rate, 4),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return result


@router.post("/model/{model_name}")
async def benchmark_model(
    model_name: str,
    iterations: int = Query(50, ge=1, le=1000),
    current_user: User = Depends(admin_required),
):
    """Benchmark a registered model's inference speed."""
    import uuid
    import numpy as np

    from main_api import app

    model = app.state.models.get(model_name)
    if not model:
        raise HTTPException(404, f"Model '{model_name}' not found")

    # Generate dummy input based on model type
    try:
        input_size = getattr(model, "input_size", 10)
    except Exception:
        input_size = 10

    latencies = []
    errors = 0

    for _ in range(iterations):
        dummy_input = np.random.randn(1, input_size).tolist()
        start = time.perf_counter()
        try:
            if hasattr(model, "predict"):
                model.predict(dummy_input)
            else:
                errors += 1
        except Exception:
            errors += 1
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    if not latencies:
        raise HTTPException(500, "All inference requests failed")

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    avg = statistics.mean(sorted_lat)

    return {
        "id": str(uuid.uuid4())[:8],
        "model_name": model_name,
        "iterations": iterations,
        "avg_latency_ms": round(avg, 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "median_latency_ms": round(statistics.median(sorted_lat), 3),
        "p95_latency_ms": round(sorted_lat[int(n * 0.95)], 3),
        "p99_latency_ms": round(sorted_lat[int(n * 0.99)], 3),
        "requests_per_second": round(1000 / avg, 1) if avg > 0 else 0,
        "error_rate": round(errors / iterations, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/history")
def benchmark_history(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    """Get recent benchmark results."""
    from database import SessionLocal

    db = SessionLocal()
    try:
        runs = (
            db.query(BenchmarkRun)
            .order_by(BenchmarkRun.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "name": r.name,
                "type": r.benchmark_type,
                "target": r.target,
                "iterations": r.iterations,
                "avg_latency_ms": r.avg_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "requests_per_second": r.requests_per_second,
                "error_rate": r.error_rate,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ]
    finally:
        db.close()


@router.get("/compare")
def compare_benchmarks(
    run_ids: str = Query(..., description="Comma-separated benchmark run IDs"),
    current_user: User = Depends(get_current_user),
):
    """Compare two or more benchmark runs side by side."""
    from database import SessionLocal

    ids = [i.strip() for i in run_ids.split(",")]
    db = SessionLocal()
    try:
        runs = db.query(BenchmarkRun).filter(BenchmarkRun.id.in_(ids)).all()
        if not runs:
            raise HTTPException(404, "No benchmark runs found for given IDs")

        comparison = []
        for r in runs:
            comparison.append({
                "id": r.id,
                "name": r.name,
                "target": r.target,
                "avg_latency_ms": r.avg_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
                "p99_latency_ms": r.p99_latency_ms,
                "requests_per_second": r.requests_per_second,
                "error_rate": r.error_rate,
            })

        # Calculate relative performance
        if len(comparison) >= 2:
            baseline = comparison[0]
            for c in comparison[1:]:
                if baseline["avg_latency_ms"] and c["avg_latency_ms"]:
                    c["relative_speed"] = round(
                        baseline["avg_latency_ms"] / c["avg_latency_ms"], 2
                    )

        return {"comparison": comparison}
    finally:
        db.close()
