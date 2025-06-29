import requests

ADMIN_TOKEN = "test_admin_token"  # Replace with a valid JWT for real tests
API_URL = "http://localhost:8000"

def test_create_user():
    resp = requests.post(f"{API_URL}/admin/users/create", data={
        "username": "testuser1",
        "email": "testuser1@example.com",
        "password": "testpass123"
    }, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert resp.status_code in (200, 201, 400)  # 400 if already exists

def test_delete_user():
    # First, create user
    create = requests.post(f"{API_URL}/admin/users/create", data={
        "username": "testuser2",
        "email": "testuser2@example.com",
        "password": "testpass123"
    }, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    if create.status_code in (200, 201):
        user_id = create.json().get("user_id")
        resp = requests.post(f"{API_URL}/admin/users/delete", data={"user_id": user_id}, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
        assert resp.status_code == 200

def test_create_model():
    resp = requests.post(f"{API_URL}/admin/models/create", data={
        "name": "testmodel1",
        "description": "A test model",
        "accuracy": 0.99
    }, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert resp.status_code in (200, 201)

def test_cancel_job():
    # This assumes a job with job_id 'testjob1' exists
    resp = requests.post(f"{API_URL}/admin/jobs/cancel", data={"job_id": "testjob1"}, headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert resp.status_code in (200, 404)
