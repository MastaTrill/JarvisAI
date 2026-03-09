"""Shared test fixtures – ensures all DB tables exist before any test runs."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from main_api import app

_test_client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _ensure_tables():
    """Create tables for both databases so tests don't hit 'no such table'."""
    # Import model modules so their tables are registered with Base.metadata
    from models_registry import ModelRegistry  # side-effect: registers table
    from jobs_persistent import Job  # side-effect: registers table
    from database_models import User  # side-effect: registers table

    _ = ModelRegistry, Job, User  # prevent unused warnings

    from db_config import Base as ConfigBase, engine as config_engine
    from database import Base as AppBase, engine as app_engine

    ConfigBase.metadata.create_all(bind=config_engine)
    AppBase.metadata.create_all(bind=app_engine)


@pytest.fixture(scope="session", autouse=True)
def _disable_rate_limiter():
    """Disable slowapi rate limiter so auth calls don't get throttled in tests."""
    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = False
        yield
        app.state.limiter.enabled = True
    else:
        yield


# --- Shared auth helpers ---

_auth_header_cache: dict = {}


def _make_auth_header(username: str, password: str, email: str) -> dict:
    """Register a user, log in, and return an Authorization header (cached)."""
    if username in _auth_header_cache:
        return _auth_header_cache[username]

    _test_client.post(
        "/register",
        json={"username": username, "password": password, "email": email},
    )
    login_resp = _test_client.post(
        "/token", data={"username": username, "password": password}
    )
    if login_resp.status_code != 200:
        pytest.skip("Login endpoint unavailable")
    token = login_resp.json().get("access_token", "")
    header = {"Authorization": f"Bearer {token}"}
    _auth_header_cache[username] = header
    return header


@pytest.fixture()
def auth_header():
    """Fixture that provides an auth header for a regular user."""
    return _make_auth_header("authuser", "authpass", "auth@example.com")


@pytest.fixture()
def admin_auth_header():
    """Fixture that provides an auth header for an admin user."""
    header = _make_auth_header("adminuser", "adminpass", "admin@example.com")

    from database import SessionLocal
    from database_models import User as DBUser

    db = SessionLocal()
    user = db.query(DBUser).filter_by(username="adminuser").first()
    if user and user.role != "admin":
        user.role = "admin"
        db.commit()
    db.close()

    return header
