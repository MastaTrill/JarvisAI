import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main_api import app

client = TestClient(app)


def test_end_to_end():
    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    # Root endpoint
    resp = client.get("/")
    assert resp.status_code == 200
