import pytest
from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)

def test_xai_explain():
    resp = client.post("/advanced/xai/explain", data={"text": "Why is the sky blue?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "explanation" in data
    assert "feature_importance" in data
    assert "counterfactuals" in data

def test_xai_explain_no_input():
    resp = client.post("/advanced/xai/explain", data={})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data