# Patch database engine/session before any other imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables are created before tests
from database_models import Base

Base.metadata.create_all(bind=engine)


import database

database.engine = engine
database.SessionLocal = TestingSessionLocal

# Ensure tables are created before any DB/model import
from database_models import Base

Base.metadata.create_all(bind=engine)

# Now import models and app
from database_models import User
from models_user import get_password_hash
from admin_dashboard import admin_required, get_db as admin_get_db
from main_api import app
from fastapi.testclient import TestClient

# Use the dedicated test session for dependency overrides.
import database


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[database.get_db] = override_get_db
app.dependency_overrides[admin_get_db] = override_get_db


def override_admin_required():
    return User(
        username="admin",
        email="admin@example.com",
        is_admin=True,
        hashed_password=get_password_hash("adminpass"),
    )


app.dependency_overrides[admin_required] = override_admin_required

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
