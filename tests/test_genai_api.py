import pytest
from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)

def test_genai_generate_text():
    resp = client.post("/advanced/genai/generate", data={"prompt": "Write a poem about AI.", "mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "text"
    assert "generated_text" in data

def test_genai_generate_image():
    resp = client.post("/advanced/genai/generate", data={"prompt": "A cat in space", "mode": "image"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "image"
    assert "image_url" in data

def test_genai_generate_no_prompt():
    resp = client.post("/advanced/genai/generate", data={"mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data