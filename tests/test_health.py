import pytest
from fastapi.testclient import TestClient
import importlib.util
import os
import sys
from fastapi.testclient import TestClient

# Dynamically import api.py as a module to get the FastAPI app
api_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api.py"))
spec = importlib.util.spec_from_file_location("api", api_path)
api_module = importlib.util.module_from_spec(spec)
sys.modules["api"] = api_module
spec.loader.exec_module(api_module)
app = api_module.app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
