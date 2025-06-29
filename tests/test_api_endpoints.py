import requests

def test_health():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_metrics():
    resp = requests.get("http://localhost:8000/metrics")
    assert resp.status_code == 200
    assert "jarvis_api_requests_total" in resp.text
