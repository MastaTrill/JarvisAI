"""Tests for JarvisAI Python SDK."""

import sys
import os
import uuid
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock

from jarvis_sdk import (
    JarvisClient,
    JarvisError,
    JarvisAuthError,
    JarvisAPIError,
    BenchmarkResult,
    ExperimentResult,
    ModelComparison,
)


class TestJarvisClientInit:
    def test_default_init(self):
        client = JarvisClient()
        assert client.base_url == "http://localhost:8000"
        assert client._token is None
        assert client._api_key is None

    def test_custom_base_url(self):
        client = JarvisClient(base_url="https://api.example.com/")
        assert client.base_url == "https://api.example.com"

    def test_init_with_token(self):
        client = JarvisClient(token="my-token")
        assert client._token == "my-token"
        assert client._session.headers["Authorization"] == "Bearer my-token"

    def test_init_with_api_key(self):
        client = JarvisClient(api_key="my-key")
        assert client._api_key == "my-key"
        assert client._session.headers["X-API-Key"] == "my-key"

    def test_context_manager(self):
        with JarvisClient() as client:
            assert client.base_url == "http://localhost:8000"
        # Session should be closed after context exit


class TestJarvisClientRequests:
    @patch("jarvis_sdk.requests.Session.request")
    def test_health(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.health()
        assert result["status"] == "ok"
        mock_request.assert_called_once()

    @patch("jarvis_sdk.requests.Session.request")
    def test_login(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "test-token-123"}
        mock_request.return_value = mock_resp

        client = JarvisClient()
        token = client.login("user", "pass")
        assert token == "test-token-123"
        assert client._token == "test-token-123"

    @patch("jarvis_sdk.requests.Session.request")
    def test_auth_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_request.return_value = mock_resp

        client = JarvisClient()
        with pytest.raises(JarvisAuthError):
            client.login("bad", "creds")

    @patch("jarvis_sdk.requests.Session.request")
    def test_api_error(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"detail": "Not found"}
        mock_request.return_value = mock_resp

        client = JarvisClient()
        with pytest.raises(JarvisAPIError) as exc_info:
            client.get_model("nonexistent")
        assert "Not found" in str(exc_info.value)

    @patch("jarvis_sdk.requests.Session.request")
    def test_list_models(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"name": "model_a"}, {"name": "model_b"}]
        mock_request.return_value = mock_resp

        client = JarvisClient()
        models = client.list_models()
        assert len(models) == 2

    @patch("jarvis_sdk.requests.Session.request")
    def test_create_experiment(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "exp123",
            "name": "test",
            "status": "draft",
            "variants": [],
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.create_experiment(
            name="test",
            variants=[{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}],
        )
        assert result["id"] == "exp123"

    @patch("jarvis_sdk.requests.Session.request")
    def test_benchmark_api(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "avg_latency_ms": 5.2,
            "min_latency_ms": 1.0,
            "max_latency_ms": 20.0,
            "median_latency_ms": 4.8,
            "p95_latency_ms": 15.0,
            "p99_latency_ms": 18.0,
            "requests_per_second": 192.3,
            "error_rate": 0.0,
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.benchmark_api("/health", iterations=100)
        assert isinstance(result, BenchmarkResult)
        assert result.avg_latency_ms == 5.2
        assert result.p95_latency_ms == 15.0
        assert result.requests_per_second == 192.3

    @patch("jarvis_sdk.requests.Session.request")
    def test_compare_models(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "comp123",
            "name": "my comparison",
            "results": {"model_a": {"accuracy": 0.95}, "model_b": {"accuracy": 0.92}},
            "winner": "model_a",
            "metric_used": "accuracy",
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.compare_models(["model_a", "model_b"])
        assert isinstance(result, ModelComparison)
        assert result.winner == "model_a"
        assert result.comparison_id == "comp123"

    @patch("jarvis_sdk.requests.Session.request")
    def test_experiment_results(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "experiment": {"name": "test", "status": "running", "winner": None},
            "results": {
                "control": {"exposures": 100, "conversions": 10, "conversion_rate": 0.1},
                "treatment": {"exposures": 100, "conversions": 15, "conversion_rate": 0.15},
            },
            "significance": {
                "z_score": 1.2,
                "p_value": 0.23,
                "significant_at_95": False,
                "lift": 50.0,
            },
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.experiment_results("exp123")
        assert isinstance(result, ExperimentResult)
        assert result.name == "test"
        assert "control" in result.results
        assert result.significance is not None

    @patch("jarvis_sdk.requests.Session.request")
    def test_system_info(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "platform": "Windows-10",
            "python_version": "3.14.5",
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client.system_info()
        assert result["platform"] == "Windows-10"

    @patch("jarvis_sdk.requests.Session.request")
    def test_list_endpoints(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "endpoints": [
                {"path": "/health", "methods": ["GET"]},
                {"path": "/models", "methods": ["GET"]},
            ]
        }
        mock_request.return_value = mock_resp

        client = JarvisClient()
        endpoints = client.list_endpoints()
        assert len(endpoints) == 2

    @patch("jarvis_sdk.requests.Session.request")
    def test_non_json_response(self, mock_request):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Not JSON")
        mock_resp.text = "plain text response"
        mock_request.return_value = mock_resp

        client = JarvisClient()
        result = client._get("/some-text-endpoint")
        assert result == "plain text response"


class TestDataclasses:
    def test_benchmark_result(self):
        r = BenchmarkResult(
            avg_latency_ms=5.0,
            min_latency_ms=1.0,
            max_latency_ms=20.0,
            median_latency_ms=4.5,
            p95_latency_ms=15.0,
            p99_latency_ms=18.0,
            requests_per_second=200.0,
            error_rate=0.01,
        )
        assert r.avg_latency_ms == 5.0
        assert r.error_rate == 0.01

    def test_experiment_result(self):
        r = ExperimentResult(
            experiment_id="exp1",
            name="test",
            status="running",
            results={},
            significance=None,
            winner=None,
        )
        assert r.experiment_id == "exp1"

    def test_model_comparison(self):
        r = ModelComparison(
            comparison_id="c1",
            name="test",
            results={},
            winner="model_a",
            metric_used="accuracy",
        )
        assert r.winner == "model_a"


class TestErrors:
    def test_jarvis_error(self):
        e = JarvisError("test error", status_code=500)
        assert str(e) == "test error"
        assert e.status_code == 500

    def test_auth_error(self):
        e = JarvisAuthError("auth failed", status_code=401)
        assert e.status_code == 401

    def test_api_error(self):
        e = JarvisAPIError("not found", status_code=404)
        assert e.status_code == 404
