"""Tests for Code Sandbox."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


def _make_user_headers():
    import uuid
    uname = f"sandboxuser_{uuid.uuid4().hex[:8]}"
    client.post(
        "/register",
        json={"username": uname, "password": "testpass123", "email": f"{uname}@test.com"},
    )
    resp = client.post("/token", data={"username": uname, "password": "testpass123"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestSandboxExecution:
    def test_execute_python_hello(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "print('Hello, Jarvis!')", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "Hello, Jarvis!" in data["output"]

    def test_execute_python_math(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "import math; print(math.sqrt(144))", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "12" in data["output"]

    def test_execute_python_error(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "1/0", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_execute_blocked_code(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "import os; os.system('rm -rf /')", "language": "python", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 400

    def test_execute_bash_echo(self):
        import platform
        if platform.system() == "Windows":
            import pytest
            pytest.skip("Bash not available on Windows")
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "echo 'Hello from bash'", "language": "bash", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Hello from bash" in data["output"]

    def test_execute_unsupported_language(self):
        headers = _make_user_headers()
        resp = client.post(
            "/sandbox/execute",
            json={"code": "print('hi')", "language": "ruby", "timeout": 10},
            headers=headers,
        )
        assert resp.status_code == 400


class TestSandboxHistory:
    def test_execution_history(self):
        headers = _make_user_headers()
        # Run something first
        client.post(
            "/sandbox/execute",
            json={"code": "print('history test')", "language": "python", "timeout": 10},
            headers=headers,
        )
        # Check history
        resp = client.get("/sandbox/history", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_supported_languages(self):
        resp = client.get("/sandbox/languages")
        assert resp.status_code == 200
        data = resp.json()
        assert "languages" in data
        lang_ids = [l["id"] for l in data["languages"]]
        assert "python" in lang_ids
        assert "bash" in lang_ids
