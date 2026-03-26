# Patch database engine/session before any other imports
import sys
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_admin.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables are created
from database_models import Base, User
from db_config import Base as ConfigBase
import models_registry  # noqa: F401  # side-effect: registers tables
import jobs_persistent  # noqa: F401  # side-effect: registers tables

_ = models_registry, jobs_persistent  # prevent unused-import warnings

Base.metadata.create_all(bind=engine)
ConfigBase.metadata.create_all(bind=engine)

from models_user import get_password_hash
from admin_dashboard import admin_required
from database import get_db as users_get_db
from db_config import get_db as jobs_get_db
from main_api import app
from fastapi.testclient import TestClient
import database


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_admin_required():
    return User(
        username="admin",
        email="admin@example.com",
        is_admin=True,
        hashed_password=get_password_hash("adminpass"),
    )


@pytest.fixture(autouse=True)
def _apply_overrides():
    """Apply dependency overrides for this module only, then clean up."""
    _orig_engine = database.engine
    _orig_session = database.SessionLocal
    database.engine = engine
    database.SessionLocal = TestingSessionLocal

    app.dependency_overrides[database.get_db] = override_get_db
    app.dependency_overrides[users_get_db] = override_get_db
    app.dependency_overrides[jobs_get_db] = override_get_db
    app.dependency_overrides[admin_required] = override_admin_required
    yield
    app.dependency_overrides.pop(database.get_db, None)
    app.dependency_overrides.pop(users_get_db, None)
    app.dependency_overrides.pop(jobs_get_db, None)
    app.dependency_overrides.pop(admin_required, None)

    database.engine = _orig_engine
    database.SessionLocal = _orig_session


client = TestClient(app)


def setup_test_users():
    db = TestingSessionLocal()
    db.query(User).delete()
    db.add_all(
        [
            User(
                username="admin",
                email="admin@example.com",
                is_admin=True,
                hashed_password=get_password_hash("adminpass"),
            ),
            User(
                username="user1",
                email="user1@example.com",
                is_admin=False,
                hashed_password=get_password_hash("user1pass"),
            ),
        ]
    )
    db.commit()
    db.close()


def test_get_admin_users():
    setup_test_users()
    response = client.get("/admin/dashboard/admin/users")
    assert response.status_code == 200
    users = response.json()
    assert any(u["username"] == "admin" for u in users)
    assert any(u["username"] == "user1" for u in users)
