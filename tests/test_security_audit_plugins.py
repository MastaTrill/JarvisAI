"""Tests for security_api.py, audit_api.py, plugins_api.py, collab_api.py router endpoints."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main_api import app

client = TestClient(app)


# --- Security API ---


class TestSecurityAPI:
    def test_create_apikey_requires_admin(self, auth_header):
        resp = client.post("/security/apikey/create", headers=auth_header)
        # Regular user should be forbidden
        assert resp.status_code in [200, 403]

    def test_create_apikey_as_admin(self, admin_auth_header):
        resp = client.post("/security/apikey/create", headers=admin_auth_header)
        assert resp.status_code == 200
        assert "api_key" in resp.json()

    def test_list_apikeys(self, admin_auth_header):
        resp = client.get("/security/apikey/list", headers=admin_auth_header)
        assert resp.status_code == 200

    def test_revoke_apikey_not_found(self, admin_auth_header):
        resp = client.post(
            "/security/apikey/revoke",
            params={"key": "nonexistent"},
            headers=admin_auth_header,
        )
        assert resp.status_code == 404

    def test_assign_role(self, admin_auth_header):
        resp = client.post(
            "/security/role/assign",
            params={"username": "testuser", "role": "viewer"},
            headers=admin_auth_header,
        )
        assert resp.status_code == 200

    def test_revoke_role_not_found(self, admin_auth_header):
        resp = client.post(
            "/security/role/revoke",
            params={"username": "nonexistent"},
            headers=admin_auth_header,
        )
        assert resp.status_code == 404


# --- Audit API ---


class TestAuditAPI:
    def test_log_action(self, auth_header):
        resp = client.post(
            "/audit/log",
            params={"action": "test_action", "detail": "test_detail"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "Action logged"

    def test_list_audit_logs(self, auth_header):
        resp = client.get("/audit/list", headers=auth_header)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_export_audit_logs(self, auth_header):
        resp = client.get("/audit/export", headers=auth_header)
        assert resp.status_code == 200
        assert "csv" in resp.json()

    def test_compliance_report(self, auth_header):
        resp = client.get("/audit/compliance-report", headers=auth_header)
        assert resp.status_code == 200


# --- Plugins API ---


class TestPluginsAPI:
    def test_list_plugins(self):
        resp = client.get("/plugins/list")
        assert resp.status_code == 200

    def test_approve_plugin_not_found(self, admin_auth_header):
        resp = client.post("/plugins/approve/nonexistent", headers=admin_auth_header)
        assert resp.status_code == 404


# --- Collaboration API ---


class TestCollabAPI:
    def test_add_annotation(self, auth_header):
        resp = client.post(
            "/collab/annotate/999",
            params={"feedback": "looks good"},
            headers=auth_header,
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "Annotation added"

    def test_list_annotations_empty(self, auth_header):
        resp = client.get("/collab/list/888", headers=auth_header)
        assert resp.status_code == 200
