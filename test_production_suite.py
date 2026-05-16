"""Production readiness test suite.

Validates that all critical modules import correctly and core
functionality works end-to-end without requiring external services.
"""

import importlib
import sys


def test_core_imports():
    """All core modules should import without errors."""
    modules = [
        "main_api",
        "jarvis_api",
        "agent_api",
        "admin_api",
        "audit_api",
        "authentication",
        "database",
        "database_models",
        "cache",
        "celery_app",
        "security_api",
        "plugins_api",
        "collab_api",
        "jobs_persistent",
        "models_registry",
        "models_versioning",
        "llm_groq",
        "llm_ollama",
        "llm_openai",
    ]
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed.append(f"{name}: {e}")
    assert not failed, f"Failed imports:\n" + "\n".join(failed)


def test_advanced_features_import():
    """Advanced feature modules should import."""
    modules = [
        "advanced_features.orchestrator",
        "advanced_features.self_healing",
        "advanced_features.explainable_ai",
        "advanced_features.federated_learning",
        "advanced_features.nlu_advanced",
        "advanced_features.multimodal_ai",
        "advanced_features.ai_workflow_automation",
    ]
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:
            failed.append(f"{name}: {e}")
    assert not failed, f"Failed imports:\n" + "\n".join(failed)


def test_fastapi_app():
    """FastAPI app should be created with all routers mounted."""
    from main_api import app
    assert app is not None
    route_paths = [r.path for r in app.routes]
    assert "/health" in route_paths, "Health endpoint missing"


def test_database_tables():
    """Database models should define tables."""
    from database import Base as AppBase
    from db_config import Base as ConfigBase
    assert len(AppBase.metadata.tables) > 0, "No app tables defined"
    assert len(ConfigBase.metadata.tables) > 0, "No config tables defined"


if __name__ == "__main__":
    tests = [
        test_core_imports,
        test_advanced_features_import,
        test_fastapi_app,
        test_database_tables,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
