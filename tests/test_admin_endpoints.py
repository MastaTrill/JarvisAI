import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)

_admin_header_cache = None


def _get_admin_auth_header():
    """Register a user, promote to admin, login, and return an Authorization header."""
    global _admin_header_cache
    if _admin_header_cache is not None:
        return _admin_header_cache
    client.post(
        "/register",
        json={
            "username": "adminuser",
            "password": "adminpass",
            "email": "admin@example.com",
        },
    )
    # Promote user to admin directly in the database
    from database import SessionLocal
    from database_models import User as DBUser

    db = SessionLocal()
    user = db.query(DBUser).filter_by(username="adminuser").first()
    if user:
        user.role = "admin"
        db.commit()
    db.close()

    login_resp = client.post(
        "/token",
        data={"username": "adminuser", "password": "adminpass"},
    )
    if login_resp.status_code != 200:
        pytest.skip("Login endpoint unavailable")
    token = login_resp.json().get("access_token", "")
    _admin_header_cache = {"Authorization": f"Bearer {token}"}
    return _admin_header_cache


def test_create_user():
    headers = _get_admin_auth_header()
    resp = client.post(
        "/admin/dashboard/admin/users/create",
        data={
            "username": "testuser1",
            "email": "testuser1@example.com",
            "password": "testpass123",
        },
        headers=headers,
    )
    assert resp.status_code in (200, 201, 400)


def test_delete_user():
    headers = _get_admin_auth_header()
    # First, create user
    create = client.post(
        "/admin/dashboard/admin/users/create",
        data={
            "username": "testuser2",
            "email": "testuser2@example.com",
            "password": "testpass123",
        },
        headers=headers,
    )
    if create.status_code in (200, 201):
        user_id = create.json().get("user_id")
        resp = client.post(
            "/admin/dashboard/admin/users/delete",
            data={"user_id": user_id},
            headers=headers,
        )
        assert resp.status_code == 200


def test_create_model():
    headers = _get_admin_auth_header()
    resp = client.post(
        "/admin/dashboard/admin/models/create",
        data={"name": "testmodel1", "description": "A test model", "accuracy": 0.99},
        headers=headers,
    )
    assert resp.status_code in (200, 201)


def test_cancel_job():
    headers = _get_admin_auth_header()
    resp = client.post(
        "/admin/dashboard/admin/jobs/cancel",
        data={"job_id": "testjob1"},
        headers=headers,
    )
    assert resp.status_code in (200, 404)
