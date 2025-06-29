import requests

def test_end_to_end():
    # Health check
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200
    # Root endpoint
    resp = requests.get("http://localhost:8000/")
    assert resp.status_code == 200
    # (Add more as needed)
