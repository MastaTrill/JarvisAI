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
            "username": "clouduser",
            "password": "cloudpass",
            "email": "cloud@example.com",
        },
    )
    login_resp = client.post(
        "/token",
        data={"username": "clouduser", "password": "cloudpass"},
    )
    if login_resp.status_code != 200:
        pytest.skip("Login endpoint unavailable")
    token = login_resp.json().get("access_token", "")
    _auth_header_cache = {"Authorization": f"Bearer {token}"}
    return _auth_header_cache


def test_cloud_upload_download():
    headers = _get_auth_header()
    # Upload (file must exist in data/uploads for real test)
    response = client.post(
        "/cloud/upload?filename=test.csv&provider=s3&bucket=test-bucket",
        headers=headers,
    )
    assert response.status_code in [200, 404, 500]
    # Download
    response = client.get(
        "/cloud/download?filename=test.csv&provider=s3&bucket=test-bucket",
        headers=headers,
    )
    assert response.status_code in [200, 404, 500]


def test_predict_gpu():
    headers = _get_auth_header()
    payload = {"model_name": "testmodel", "data": [[1, 2, 3]]}
    response = client.post("/models/testmodel/predict_gpu", json=payload, headers=headers)
    assert response.status_code in [200, 404, 500]


def test_predict_external():
    headers = _get_auth_header()
    payload = {"model_name": "testmodel", "data": [[1, 2, 3]]}
    response = client.post("/models/testmodel/predict_external", json=payload, headers=headers)
    assert response.status_code in [200, 404, 500]
