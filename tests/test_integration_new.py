"""Integration tests for new features: RAG, Sandbox, Events, Personality, Analytics."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def _make_user_headers():
    import uuid
    uname = f"inttest_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _make_admin_headers():
    import uuid
    uname = f"intadmin_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    from database import SessionLocal
    from database_models import User as DBUser
    db = SessionLocal()
    user = db.query(DBUser).filter_by(username=uname).first()
    user.role = "admin"
    user.is_admin = True
    db.commit()
    db.close()
    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestRAGIntegration:
    """Test full RAG flow: upload → list → query → delete."""

    def test_full_rag_flow(self):
        headers = _make_user_headers()

        # 1. Upload a document
        doc_content = b"Machine learning is a subset of artificial intelligence. Deep learning is a subset of machine learning. Neural networks are the foundation of deep learning."
        upload_resp = client.post(
            "/rag/upload",
            files={"file": ("ml_guide.txt", doc_content, "text/plain")},
            headers=headers,
        )
        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["document_id"]
        assert upload_resp.json()["status"] == "ready"

        # 2. List documents
        list_resp = client.get("/rag/documents", headers=headers)
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert any(d["id"] == doc_id for d in docs)

        # 3. Query the document
        query_resp = client.post(
            "/rag/query",
            json={"query": "What is deep learning?", "top_k": 3},
            headers=headers,
        )
        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert len(results) >= 1
        # The result should mention deep learning
        combined = " ".join(r["content"] for r in results).lower()
        assert "deep learning" in combined

        # 4. Delete the document
        delete_resp = client.delete(f"/rag/documents/{doc_id}", headers=headers)
        assert delete_resp.status_code == 200

    def test_rag_multiple_documents(self):
        headers = _make_user_headers()

        # Upload multiple documents
        docs = [
            ("python.txt", b"Python is a high-level programming language. It was created by Guido van Rossum."),
            ("javascript.txt", b"JavaScript is the language of the web. It runs in browsers and on servers."),
            ("rust.txt", b"Rust is a systems programming language focused on safety and performance."),
        ]
        for filename, content in docs:
            resp = client.post(
                "/rag/upload",
                files={"file": (filename, content, "text/plain")},
                headers=headers,
            )
            assert resp.status_code == 200

        # Query for Python
        query_resp = client.post(
            "/rag/query",
            json={"query": "Who created Python?", "top_k": 3},
            headers=headers,
        )
        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert len(results) >= 1

    def test_rag_reindex(self):
        headers = _make_user_headers()
        # Upload
        client.post(
            "/rag/upload",
            files={"file": ("reindex_test.txt", b"Test content for reindexing.", "text/plain")},
            headers=headers,
        )
        # Reindex
        resp = client.post("/rag/reindex", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] >= 1


class TestSandboxIntegration:
    """Test full sandbox flow: execute → history → languages."""

    def test_full_sandbox_flow(self):
        headers = _make_user_headers()

        # 1. Execute Python code
        exec_resp = client.post(
            "/sandbox/execute",
            json={"code": "print('Integration test'); x = 42; print(f'Answer: {x}')", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert exec_resp.status_code == 200
        data = exec_resp.json()
        assert data["status"] == "success"
        assert "Integration test" in data["output"]
        assert "Answer: 42" in data["output"]
        assert data["exit_code"] == 0

        # 2. Check history
        history_resp = client.get("/sandbox/history", headers=headers)
        assert history_resp.status_code == 200
        history = history_resp.json()
        assert len(history) >= 1
        assert history[0]["language"] == "python"

        # 3. Check supported languages
        lang_resp = client.get("/sandbox/languages")
        assert lang_resp.status_code == 200
        langs = lang_resp.json()["languages"]
        lang_ids = [l["id"] for l in langs]
        assert "python" in lang_ids

    def test_sandbox_error_handling(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "raise ValueError('test error')", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["exit_code"] != 0

    def test_sandbox_blocked_code(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "__import__('os').system('ls')", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 400


class TestEventsIntegration:
    """Test event publishing and history."""

    def test_publish_and_history(self):
        headers = _make_user_headers()

        # 1. Publish events
        for i in range(3):
            resp = client.post(
                "/events/publish",
                json={"channel": "test", "event_type": "test_event", "data": {"index": i}},
                headers=headers,
            )
            assert resp.status_code == 200

        # 2. Check history
        history_resp = client.get("/events/history/test?limit=10", headers=headers)
        assert history_resp.status_code == 200
        events = history_resp.json()["events"]
        assert len(events) >= 3

    def test_list_channels(self):
        headers = _make_user_headers()
        # Publish to a channel first
        client.post(
            "/events/publish",
            json={"channel": "my_channel", "event_type": "test", "data": {}},
            headers=headers,
        )
        resp = client.get("/events/channels", headers=headers)
        assert resp.status_code == 200
        channels = resp.json()["channels"]
        assert "my_channel" in channels


class TestPersonalityIntegration:
    """Test full personality flow: create → list → activate → delete."""

    def test_full_personality_flow(self):
        headers = _make_user_headers()

        # 1. Create personality
        create_resp = client.post(
            "/agent/config/personality",
            json={
                "name": "test-personality",
                "display_name": "Test Bot",
                "tone": "friendly",
                "system_prompt": "You are a helpful test assistant.",
                "expertise": "testing,debugging",
            },
            headers=headers,
        )
        assert create_resp.status_code == 200

        # 2. List personalities
        list_resp = client.get("/agent/config/personalities", headers=headers)
        assert list_resp.status_code == 200
        personalities = list_resp.json()
        assert any(p["name"] == "test-personality" for p in personalities)

        # 3. Get specific personality
        get_resp = client.get("/agent/config/personality/test-personality", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["display_name"] == "Test Bot"
        assert get_resp.json()["tone"] == "friendly"

        # 4. Activate personality
        activate_resp = client.post("/agent/config/personality/test-personality/activate", headers=headers)
        assert activate_resp.status_code == 200

        # 5. Verify it's the default
        list_resp2 = client.get("/agent/config/personalities", headers=headers)
        active = [p for p in list_resp2.json() if p["is_default"]]
        assert len(active) == 1
        assert active[0]["name"] == "test-personality"

        # 6. Delete personality
        delete_resp = client.delete("/agent/config/personality/test-personality", headers=headers)
        assert delete_resp.status_code == 200

    def test_model_routing(self):
        headers = _make_user_headers()

        # 1. Create a model route
        route_resp = client.post(
            "/agent/config/model-route",
            json={
                "name": "code-route",
                "task_type": "code",
                "provider": "ollama",
                "model_name": "codellama",
                "priority": 10,
            },
            headers=headers,
        )
        assert route_resp.status_code == 200

        # 2. List routes
        list_resp = client.get("/agent/config/model-routes", headers=headers)
        assert list_resp.status_code == 200
        routes = list_resp.json()
        assert any(r["name"] == "code-route" for r in routes)

        # 3. Resolve model for task
        resolve_resp = client.get("/agent/config/model-route/resolve?task_type=code", headers=headers)
        assert resolve_resp.status_code == 200
        assert resolve_resp.json()["model"] == "codellama"


class TestAnalyticsIntegration:
    """Test API analytics endpoints."""

    def test_analytics_overview(self):
        headers = _make_admin_headers()
        # The analytics overview should return valid structure even with 0 requests
        resp = client.get("/analytics/overview", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "requests_last_24h" in data
        assert "unique_users" in data
        assert "uptime_seconds" in data
        assert "top_endpoints" in data

    def test_endpoint_analytics(self):
        headers = _make_admin_headers()
        resp = client.get("/analytics/endpoints?limit=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data

    def test_timeline_analytics(self):
        headers = _make_admin_headers()
        resp = client.get("/analytics/timeline?hours=1", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "timeline" in data
