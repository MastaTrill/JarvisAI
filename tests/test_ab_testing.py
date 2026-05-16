"""Tests for A/B Testing framework."""

import sys
import os
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def _make_admin_headers():
    """Create an admin user and return auth headers."""
    uname = f"abadmin_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    # Make admin
    from database import SessionLocal
    from database_models import User as DBUser

    db = SessionLocal()
    user = db.query(DBUser).filter_by(username=uname).first()
    user.role = "admin"
    user.is_admin = True
    db.commit()
    db.close()

    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestExperimentCreation:
    def test_create_experiment(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/experiments/create",
            json={
                "name": f"test_exp_{uuid.uuid4().hex[:6]}",
                "description": "Test experiment",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "variant_b", "weight": 0.5},
                ],
                "primary_metric": "conversion",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"].startswith("test_exp_")
        assert data["status"] == "draft"
        assert len(data["variants"]) == 2

    def test_create_experiment_requires_two_variants(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/experiments/create",
            json={
                "name": "bad_exp",
                "variants": [{"name": "only_one", "weight": 1.0}],
            },
            headers=headers,
        )
        assert resp.status_code == 400

    def test_create_experiment_weights_must_sum_to_one(self):
        headers = _make_admin_headers()
        resp = client.post(
            "/experiments/create",
            json={
                "name": "bad_exp",
                "variants": [
                    {"name": "a", "weight": 0.3},
                    {"name": "b", "weight": 0.3},
                ],
            },
            headers=headers,
        )
        assert resp.status_code == 400


class TestExperimentLifecycle:
    def test_start_and_stop_experiment(self):
        headers = _make_admin_headers()

        # Create
        create_resp = client.post(
            "/experiments/create",
            json={
                "name": f"lifecycle_exp_{uuid.uuid4().hex[:6]}",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ],
            },
            headers=headers,
        )
        exp_id = create_resp.json()["id"]

        # Start
        start_resp = client.post(f"/experiments/{exp_id}/start", headers=headers)
        assert start_resp.status_code == 200

        # Stop with winner
        stop_resp = client.post(
            f"/experiments/{exp_id}/stop?winner=treatment", headers=headers
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["winner"] == "treatment"

    def test_list_experiments(self):
        headers = _make_admin_headers()
        resp = client.get("/experiments/list", headers=headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_experiments_by_status(self):
        headers = _make_admin_headers()
        resp = client.get("/experiments/list?status=draft", headers=headers)
        assert resp.status_code == 200
        for exp in resp.json():
            assert exp["status"] == "draft"


class TestExperimentResults:
    def test_get_results(self):
        headers = _make_admin_headers()

        # Create and start
        create_resp = client.post(
            "/experiments/create",
            json={
                "name": f"results_exp_{uuid.uuid4().hex[:6]}",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ],
            },
            headers=headers,
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/experiments/{exp_id}/start", headers=headers)

        # Get results
        resp = client.get(f"/experiments/{exp_id}/results", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "control" in data["results"]
        assert "treatment" in data["results"]

    def test_track_event(self):
        headers = _make_admin_headers()

        # Create and start
        create_resp = client.post(
            "/experiments/create",
            json={
                "name": f"track_exp_{uuid.uuid4().hex[:6]}",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ],
            },
            headers=headers,
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/experiments/{exp_id}/start", headers=headers)

        # Track conversion event (needs auth)
        user_headers = _make_admin_headers()
        resp = client.post(
            "/experiments/track",
            json={
                "experiment_id": exp_id,
                "event_type": "conversion",
                "event_name": "signup",
                "event_value": 1.0,
            },
            headers=user_headers,
        )
        assert resp.status_code == 200

    def test_get_variant_assignment(self):
        headers = _make_admin_headers()

        # Create and start
        create_resp = client.post(
            "/experiments/create",
            json={
                "name": f"variant_exp_{uuid.uuid4().hex[:6]}",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ],
            },
            headers=headers,
        )
        exp_id = create_resp.json()["id"]
        client.post(f"/experiments/{exp_id}/start", headers=headers)

        # Get variant (needs auth)
        resp = client.get(f"/experiments/{exp_id}/variant", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["variant"] in ("control", "treatment")
        assert data["experiment_id"] == exp_id
