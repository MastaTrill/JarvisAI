"""Shared test fixtures – ensures all DB tables exist before any test runs."""

import sys
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_TEMP_DIR = PROJECT_ROOT / "scratch" / "pytest-temp"
PROJECT_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Disable Redis for tests to avoid connection errors
os.environ["REDIS_URL"] = ""

# Load .env file if it exists
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                # Handle values that might contain = by splitting only once from the left
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key, value = parts
                    # Remove surrounding quotes if present
                    value = value.strip("\"'")
                    key = key.strip()
                    # Skip Redis URL for tests to avoid connection errors
                    if key == "REDIS_URL":
                        continue
                    os.environ[key] = value

# Keep pytest and tempfile usage inside the repo so Windows temp directory
# permission issues do not break the suite.
os.environ["TMP"] = str(PROJECT_TEMP_DIR)
os.environ["TEMP"] = str(PROJECT_TEMP_DIR)
os.environ["TMPDIR"] = str(PROJECT_TEMP_DIR)
tempfile.tempdir = str(PROJECT_TEMP_DIR)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

# Global app reference, lazily loaded on first fixture use
_app = None
_test_client = None


def _get_app():
    """Lazily load the FastAPI app on first access."""
    global _app
    if _app is None:
        from main_api import app as loaded_app

        _app = loaded_app
    return _app


def _get_test_client():
    """Lazily load the TestClient on first access."""
    global _test_client
    if _test_client is None:
        app = _get_app()
        _test_client = TestClient(app)
    return _test_client


@pytest.fixture(scope="session", autouse=True)
def _ensure_tables():
    """Create tables for both databases so tests don't hit 'no such table'."""
    # Import model modules so their tables are registered with Base.metadata
    from src.ml.models_registry import ModelRegistry  # side-effect: registers table
    from src.ml.jobs_persistent import Job  # side-effect: registers table
    from src.infra.database_models import User  # side-effect: registers table
    from src.ml.models_versioning import ModelVersion  # side-effect: registers table
    from ab_testing import ABExperiment, ABEvent  # side-effect: registers tables
    from benchmarking import BenchmarkRun  # side-effect: registers table
    from src.ml.model_comparison import ModelComparison  # side-effect: registers table

    _ = ModelRegistry, Job, User, ModelVersion, ABExperiment, ABEvent, BenchmarkRun, ModelComparison

    from src.infra.db_config import Base as ConfigBase, engine as config_engine
    from src.infra.database import Base as AppBase, engine as app_engine

    ConfigBase.metadata.create_all(bind=config_engine)
    AppBase.metadata.create_all(bind=app_engine)

    # ab_testing and benchmarking use ConfigBase for models but get_db from database.py,
    # so their tables must also exist on the app engine.
    from sqlalchemy import MetaData
    for table in ConfigBase.metadata.sorted_tables:
        if table.key not in {t.key for t in AppBase.metadata.sorted_tables}:
            table.create(bind=app_engine, checkfirst=True)


@pytest.fixture(scope="session", autouse=True)
def _disable_rate_limiter():
    """Disable slowapi rate limiter so auth calls don't get throttled in tests."""
    app = _get_app()
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

    client = _get_test_client()
    client.post(
        "/register",
        json={"username": username, "password": password, "email": email},
    )
    login_resp = client.post(
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

    from src.infra.database import SessionLocal
    from src.infra.database_models import User as DBUser

    db = SessionLocal()
    user = db.query(DBUser).filter_by(username="adminuser").first()
    if user and user.role != "admin":
        user.role = "admin"
        db.commit()
    db.close()

    return header
