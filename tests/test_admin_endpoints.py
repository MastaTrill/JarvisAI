import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)


def test_create_user(admin_auth_header):
    resp = client.post(
        "/admin/dashboard/admin/users/create",
        data={
            "username": "testuser1",
            "email": "testuser1@example.com",
            "password": "testpass123",
        },
        headers=admin_auth_header,
    )
    assert resp.status_code in (200, 201, 400)


def test_delete_user(admin_auth_header):
    # First, create user
    create = client.post(
        "/admin/dashboard/admin/users/create",
        data={
            "username": "testuser2",
            "email": "testuser2@example.com",
            "password": "testpass123",
        },
        headers=admin_auth_header,
    )
    if create.status_code in (200, 201):
        user_id = create.json().get("user_id")
        resp = client.post(
            "/admin/dashboard/admin/users/delete",
            data={"user_id": user_id},
            headers=admin_auth_header,
        )
        assert resp.status_code == 200


def test_create_model(admin_auth_header):
    resp = client.post(
        "/admin/dashboard/admin/models/create",
        data={"name": "testmodel1", "description": "A test model", "accuracy": 0.99},
        headers=admin_auth_header,
    )
    assert resp.status_code in (200, 201)


def test_cancel_job(admin_auth_header):
    resp = client.post(
        "/admin/dashboard/admin/jobs/cancel",
        data={"job_id": "testjob1"},
        headers=admin_auth_header,
    )
    assert resp.status_code in (200, 404)
