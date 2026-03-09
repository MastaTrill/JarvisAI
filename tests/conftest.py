"""Shared test fixtures – ensures all DB tables exist before any test runs."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_tables():
    """Create tables for both databases so tests don't hit 'no such table'."""
    # Import model modules so their tables are registered with Base.metadata
    import models_registry  # noqa: F401
    import jobs_persistent  # noqa: F401
    import database_models  # noqa: F401

    from db_config import Base as ConfigBase, engine as config_engine
    from database import Base as AppBase, engine as app_engine

    ConfigBase.metadata.create_all(bind=config_engine)
    AppBase.metadata.create_all(bind=app_engine)


@pytest.fixture(scope="session", autouse=True)
def _disable_rate_limiter():
    """Disable slowapi rate limiter so auth calls don't get throttled in tests."""
    from main_api import app

    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = False
        yield
        app.state.limiter.enabled = True
    else:
        yield
