import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "jarvis_api_requests_total" in response.text


def test_list_models_registry():
    response = client.get("/models/registry")
    assert response.status_code == 200


def test_start_job():
    response = client.post("/jobs/start", json={"duration": 1})
    assert response.status_code == 200


def test_job_status():
    # This test assumes a job_id is available; in real tests, chain with start_job
    response = client.get("/jobs/status/invalid-job-id")
    assert response.status_code == 200


def test_cancel_job():
    response = client.post("/jobs/cancel/invalid-job-id")
    assert response.status_code == 200
