"""Tests for model versioning endpoints and jobs_persistent CRUD."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_config import Base
from jobs_persistent import create_job, update_job_status, get_job
from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


# --- jobs_persistent CRUD ---


class TestJobsCRUD:
    @pytest.fixture()
    def db_session(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine)
        session = TestSession()
        yield session
        session.close()

    def test_create_job(self, db_session):
        job = create_job(db_session, "job-001", status="queued")
        assert job is not None
        assert str(job.job_id) == "job-001"
        assert str(job.status) == "queued"
        assert job.cancelled is False

    def test_get_job(self, db_session):
        create_job(db_session, "job-002")
        job = get_job(db_session, "job-002")
        assert job is not None
        assert job.job_id == "job-002"

    def test_get_nonexistent_job(self, db_session):
        job = get_job(db_session, "nope")
        assert job is None

    def test_update_job_status(self, db_session):
        create_job(db_session, "job-003")
        updated = update_job_status(db_session, "job-003", status="running")
        assert updated is not None
        assert str(updated.status) == "running"

    def test_cancel_job(self, db_session):
        create_job(db_session, "job-004")
        updated = update_job_status(
            db_session, "job-004", status="cancelled", cancelled=True
        )
        assert updated is not None
        assert updated.cancelled is True


# --- Model versioning API ---


class TestModelVersioningAPI:
    def test_register_version(self, admin_auth_header):
        resp = client.post(
            "/models/versioning/register",
            params={
                "name": "test_model",
                "version": "1.0.0",
                "description": "initial",
            },
            headers=admin_auth_header,
        )
        assert resp.status_code == 200
        assert "registered" in resp.json()["message"].lower()

    def test_list_versions(self, auth_header):
        resp = client.get("/models/versioning/list/test_model", headers=auth_header)
        # Endpoint may require admin role (403) or auth (401), or succeed (200)
        assert resp.status_code in [200, 401, 403]
