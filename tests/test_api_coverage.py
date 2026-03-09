"""Tests for core API endpoints — register, login, /me, model registry, jobs, GDPR."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


class TestRegistration:
    def test_register_new_user(self):
        uname = f"newuser_{uuid.uuid4().hex[:8]}"
        resp = client.post(
            "/register",
            json={
                "username": uname,
                "password": "testpass123",
                "email": f"{uname}@test.com",
            },
        )
        assert resp.status_code == 200
        assert "registered" in resp.json()["msg"].lower()

    def test_register_duplicate_user(self):
        uname = f"dupuser_{uuid.uuid4().hex[:8]}"
        payload = {
            "username": uname,
            "password": "testpass123",
            "email": f"{uname}@test.com",
        }
        client.post("/register", json=payload)
        resp = client.post("/register", json=payload)
        assert resp.status_code == 400


class TestLogin:
    def test_login_success(self):
        uname = f"loginuser_{uuid.uuid4().hex[:8]}"
        client.post(
            "/register",
            json={
                "username": uname,
                "password": "testpass123",
                "email": f"{uname}@test.com",
            },
        )
        resp = client.post(
            "/token", data={"username": uname, "password": "testpass123"}
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password(self):
        resp = client.post(
            "/token", data={"username": "nonexistent", "password": "wrong"}
        )
        assert resp.status_code == 401

    def test_token_type_is_bearer(self):
        uname = f"bearer_{uuid.uuid4().hex[:8]}"
        client.post(
            "/register",
            json={
                "username": uname,
                "password": "testpass123",
                "email": f"{uname}@test.com",
            },
        )
        resp = client.post(
            "/token", data={"username": uname, "password": "testpass123"}
        )
        assert resp.json()["token_type"] == "bearer"


class TestMe:
    def test_me_authenticated(self, auth_header):
        resp = client.get("/me", headers=auth_header)
        assert resp.status_code == 200
        assert "username" in resp.json()

    def test_me_unauthenticated(self):
        resp = client.get("/me")
        assert resp.status_code in [401, 403]


class TestModelRegistry:
    def test_register_model(self, auth_header):
        resp = client.post(
            "/models/register",
            params={"model_name": "test_model", "description": "a test"},
            headers=auth_header,
        )
        assert resp.status_code == 200

    def test_list_registered_models(self, auth_header):
        resp = client.get("/models/registry", headers=auth_header)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_activate_model_not_found(self, auth_header):
        resp = client.post(
            "/models/activate",
            params={"model_name": "nonexistent_xyz"},
            headers=auth_header,
        )
        assert resp.status_code == 404


class TestJobs:
    def test_start_job(self, auth_header):
        resp = client.post("/jobs/start", params={"duration": 1}, headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_job_status_not_found(self, auth_header):
        resp = client.get("/jobs/status/nonexistent", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json()["status"] == "not found"

    def test_cancel_job(self, auth_header):
        # Start a job first
        start = client.post("/jobs/start", params={"duration": 60}, headers=auth_header)
        job_id = start.json()["job_id"]
        resp = client.post(f"/jobs/cancel/{job_id}", headers=auth_header)
        assert resp.status_code == 200


class TestGDPR:
    def test_anonymize_nonexistent_user(self, auth_header):
        resp = client.post("/gdpr/anonymize/nonexistent_user_xyz", headers=auth_header)
        assert resp.status_code in [200, 404]

    def test_delete_nonexistent_user(self, auth_header):
        resp = client.delete("/gdpr/delete/nonexistent_user_xyz", headers=auth_header)
        assert resp.status_code in [200, 404]


class TestSystemEndpoints:
    def test_system_metrics(self):
        resp = client.get("/api/system/metrics")
        assert resp.status_code == 200
        assert "cpu_usage" in resp.json()

    def test_system_metrics_history(self):
        resp = client.get("/api/system/metrics/history")
        assert resp.status_code == 200
        assert "metrics" in resp.json()

    def test_training_status_not_found(self):
        resp = client.get("/api/training/status/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_models_list_detailed(self):
        resp = client.get("/api/models/list")
        assert resp.status_code == 200
        assert "models" in resp.json()
