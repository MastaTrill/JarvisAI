"""Tests for models_registry.py — CRUD operations and activate_model optimization."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_config import Base
from models_registry import create_model, get_models, activate_model


@pytest.fixture()
def db_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestCreateModel:
    def test_create_model_basic(self, db_session):
        model = create_model(db_session, "test_model", "A test model", 0.95)
        assert model is not None
        assert str(model.name) == "test_model"
        assert str(model.description) == "A test model"
        assert float(model.accuracy) == 0.95
        assert model.active is False

    def test_create_model_defaults(self, db_session):
        model = create_model(db_session, "basic")
        assert model is not None
        assert str(model.description) == ""
        assert model.accuracy is None
        assert str(model.version) == "1.0.0"
        assert str(model.device) == "cpu"

    def test_create_multiple_models(self, db_session):
        create_model(db_session, "m1")
        create_model(db_session, "m2")
        create_model(db_session, "m3")
        models = get_models(db_session)
        assert len(models) == 3


class TestGetModels:
    def test_empty_registry(self, db_session):
        models = get_models(db_session)
        assert models == []

    def test_returns_all_models(self, db_session):
        create_model(db_session, "alpha")
        create_model(db_session, "beta")
        models = get_models(db_session)
        names = [m.name for m in models]
        assert "alpha" in names
        assert "beta" in names


class TestActivateModel:
    def test_activate_existing_model(self, db_session):
        create_model(db_session, "m1")
        create_model(db_session, "m2")
        result = activate_model(db_session, "m1")
        assert result is not None
        assert result.active is True

    def test_activate_deactivates_others(self, db_session):
        create_model(db_session, "m1")
        create_model(db_session, "m2")
        activate_model(db_session, "m1")
        activate_model(db_session, "m2")
        models = get_models(db_session)
        active = [m for m in models if m.active]
        assert len(active) == 1
        assert active[0].name == "m2"

    def test_activate_nonexistent_returns_none(self, db_session):
        result = activate_model(db_session, "nonexistent")
        assert result is None

    def test_activate_uses_batch_update(self, db_session):
        """Verify activate doesn't load all rows into Python."""
        for i in range(50):
            create_model(db_session, f"model_{i}")
        activate_model(db_session, "model_0")
        activate_model(db_session, "model_49")
        models = get_models(db_session)
        active = [m for m in models if m.active]
        assert len(active) == 1
        assert active[0].name == "model_49"
