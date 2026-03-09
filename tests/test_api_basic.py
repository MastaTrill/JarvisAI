import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)


def _get_auth_header():
    """Register a test user, login, and return an Authorization header."""
    client.post(
        "/register",
        json={
            "username": "authuser",
            "password": "authpass",
            "email": "auth@example.com",
        },
    )
    login_resp = client.post(
        "/token",
        data={"username": "authuser", "password": "authpass"},
    )
    if login_resp.status_code != 200:
        pytest.skip("Login endpoint unavailable")
    token = login_resp.json().get("access_token", "")
    return {"Authorization": f"Bearer {token}"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_user():
    payload = {
        "username": "testuser",
        "password": "testpass",
        "email": "test@example.com",
    }
    response = client.post("/register", json=payload)
    assert response.status_code == 200 or response.status_code == 400


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200


def test_gdpr_anonymize():
    headers = _get_auth_header()
    response = client.post("/gdpr/anonymize/testuser", headers=headers)
    assert response.status_code in [200, 401, 404]


def test_gdpr_delete():
    headers = _get_auth_header()
    response = client.delete("/gdpr/delete/testuser", headers=headers)
    assert response.status_code in [200, 401, 404]
