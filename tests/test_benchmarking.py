"""Tests for Performance Benchmarking system."""

import sys
import os
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def _make_admin_headers():
    uname = f"benchadmin_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    from database import SessionLocal
    from database_models import User as DBUser

    db = SessionLocal()
    user = db.query(DBUser).filter_by(username=uname).first()
    user.role = "admin"
    user.is_admin = True
    db.commit()
    db.close()

    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestAPIBenchmark:
    def test_benchmark_health_endpoint(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/benchmarks/api?endpoint=/health&iterations=10",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "avg_latency_ms" in data
        assert "p95_latency_ms" in data
        assert "p99_latency_ms" in data
        assert "requests_per_second" in data
        assert "error_rate" in data
        assert data["iterations"] == 10

    def test_benchmark_returns_valid_latency_values(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/benchmarks/api?endpoint=/health&iterations=20",
            headers=headers,
        )
        data = resp.json()
        assert data["min_latency_ms"] <= data["avg_latency_ms"]
        assert data["avg_latency_ms"] <= data["max_latency_ms"]
        assert data["p95_latency_ms"] <= data["p99_latency_ms"]
        assert data["error_rate"] == 0.0

    def test_benchmark_system_snapshot(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/benchmarks/api?endpoint=/health&iterations=5",
            headers=headers,
        )
        data = resp.json()
        assert "system" in data
        assert "cpu_percent" in data["system"]
        assert "memory_percent" in data["system"]


class TestBenchmarkHistory:
    def test_benchmark_history(self):
        headers = _make_admin_headers()
        resp = client.get("/benchmarks/history?limit=10", headers=headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_benchmark_history_default_limit(self):
        headers = _make_admin_headers()
        resp = client.get("/benchmarks/history", headers=headers)
        assert resp.status_code == 200
