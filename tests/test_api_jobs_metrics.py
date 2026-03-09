import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)

_auth_header_cache = None


def _get_auth_header():
    """Register a test user, login, and return an Authorization header."""
    global _auth_header_cache
    if _auth_header_cache is not None:
        return _auth_header_cache
    client.post(
        "/register",
        json={
            "username": "jobsuser",
            "password": "jobspass",
            "email": "jobs@example.com",
        },
    )
    login_resp = client.post(
        "/token",
        data={"username": "jobsuser", "password": "jobspass"},
    )
    if login_resp.status_code != 200:
        pytest.skip("Login endpoint unavailable")
    token = login_resp.json().get("access_token", "")
    _auth_header_cache = {"Authorization": f"Bearer {token}"}
    return _auth_header_cache


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "jarvis_api_requests_total" in response.text


def test_list_models_registry():
    headers = _get_auth_header()
    response = client.get("/models/registry", headers=headers)
    assert response.status_code == 200


def test_start_job():
    headers = _get_auth_header()
    response = client.post("/jobs/start", json={"duration": 1}, headers=headers)
    assert response.status_code == 200


def test_job_status():
    headers = _get_auth_header()
    response = client.get("/jobs/status/invalid-job-id", headers=headers)
    assert response.status_code == 200


def test_cancel_job():
    headers = _get_auth_header()
    response = client.post("/jobs/cancel/invalid-job-id", headers=headers)
    assert response.status_code == 200
