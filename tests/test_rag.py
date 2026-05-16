"""Tests for Document RAG system."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def _make_user_headers():
    import uuid
    uname = f"raguser_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestRAGUpload:
    def test_upload_text_file(self):
        headers = _make_user_headers()
        content = b"This is a test document about artificial intelligence and machine learning."
        resp = client.post(
            "/rag/upload",
            files={"file": ("test_doc.txt", content, "text/plain")},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["chunks"] >= 1

    def test_upload_markdown_file(self):
        headers = _make_user_headers()
        content = b"# Hello\n\nThis is a **markdown** document.\n\n## Section 2\n\nSome content here."
        resp = client.post(
            "/rag/upload",
            files={"file": ("readme.md", content, "text/markdown")},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_upload_unsupported_file_type(self):
        headers = _make_user_headers()
        resp = client.post(
            "/rag/upload",
            files={"file": ("image.png", b"\x89PNG", "image/png")},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_list_documents(self):
        headers = _make_user_headers()
        resp = client.get("/rag/documents", headers=headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestRAGQuery:
    def test_query_after_upload(self):
        headers = _make_user_headers()
        # Upload a document
        content = b"Python is a programming language. JavaScript is used for web development."
        client.post(
            "/rag/upload",
            files={"file": ("langs.txt", content, "text/plain")},
            headers=headers,
        )
        # Query it
        resp = client.post(
            "/rag/query",
            json={"query": "What is Python?", "top_k": 3},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_query_no_documents(self):
        headers = _make_user_headers()
        resp = client.post(
            "/rag/query",
            json={"query": "xyznonexistent12345", "top_k": 3},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data


class TestRAGDelete:
    def test_delete_document(self):
        headers = _make_user_headers()
        # Upload
        content = b"Test content for deletion."
        upload_resp = client.post(
            "/rag/upload",
            files={"file": ("delete_me.txt", content, "text/plain")},
            headers=headers,
        )
        doc_id = upload_resp.json()["document_id"]
        # Delete
        resp = client.delete(f"/rag/documents/{doc_id}", headers=headers)
        assert resp.status_code == 200

    def test_delete_nonexistent_document(self):
        headers = _make_user_headers()
        resp = client.delete("/rag/documents/99999", headers=headers)
        assert resp.status_code == 404
