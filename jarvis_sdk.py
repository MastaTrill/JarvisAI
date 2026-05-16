"""
JarvisAI Python SDK

A clean client library for interacting with the JarvisAI API.
Supports authentication, model management, experiments, benchmarks, and more.

Usage:
    from jarvis_sdk import JarvisClient

    client = JarvisClient(base_url="http://localhost:8000")
    client.login("username", "password")

    # Health check
    health = client.health()

    # List models
    models = client.list_models()

    # Create experiment
    exp = client.create_experiment(
        name="button_color_test",
        variants=[
            {"name": "control", "weight": 0.5},
            {"name": "variant_b", "weight": 0.5}
        ]
    )

    # Run benchmark
    result = client.benchmark_api("/health", iterations=100)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    import requests
except ImportError:
    raise ImportError("jarvis_sdk requires 'requests'. Install it with: pip install requests")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_rate: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of an A/B experiment."""
    experiment_id: str
    name: str
    status: str
    results: Dict[str, Any]
    significance: Optional[Dict[str, Any]]
    winner: Optional[str]


@dataclass
class ModelComparison:
    """Result of a model comparison."""
    comparison_id: str
    name: str
    results: Dict[str, Any]
    winner: Optional[str]
    metric_used: str


class JarvisError(Exception):
    """Base exception for Jarvis SDK errors."""
    def __init__(self, message: str, status_code: int = 0, response: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class JarvisAuthError(JarvisError):
    """Authentication error."""
    pass


class JarvisAPIError(JarvisError):
    """API returned an error response."""
    pass


class JarvisClient:
    """Client for the JarvisAI API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._token = token
        self._api_key = api_key

        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict] = None,
    ) -> Any:
        """Make an API request and return parsed JSON."""
        url = f"{self.base_url}{path}"
        resp = self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            data=data,
            headers=headers,
            timeout=self.timeout,
        )

        if resp.status_code == 401:
            raise JarvisAuthError("Authentication failed", resp.status_code, resp)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise JarvisAPIError(
                f"API error ({resp.status_code}): {detail}",
                resp.status_code,
                resp,
            )

        try:
            return resp.json()
        except Exception:
            return resp.text

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json_data: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
        return self._request("POST", path, json_data=json_data, params=params)

    # --- Authentication ---

    def login(self, username: str, password: str) -> str:
        """Login and store the JWT token."""
        resp = self._request(
            "POST",
            "/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self._token = resp.get("access_token")
        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"
        return self._token

    def register(self, username: str, password: str, email: str) -> Dict:
        """Register a new user."""
        return self._post("/register", json_data={
            "username": username,
            "password": password,
            "email": email,
        })

    # --- Health ---

    def health(self) -> Dict[str, Any]:
        """Basic health check."""
        return self._get("/health")

    def health_dashboard(self) -> Dict[str, Any]:
        """Detailed health dashboard with system stats."""
        return self._get("/health/dashboard")

    # --- Models ---

    def list_models(self) -> List[Dict]:
        """List all registered models."""
        return self._get("/models")

    def get_model(self, name: str) -> Dict:
        """Get a specific model by name."""
        return self._get(f"/models/{name}")

    def register_model(
        self,
        name: str,
        description: str = "",
        accuracy: Optional[float] = None,
    ) -> Dict:
        """Register a new model (admin only)."""
        return self._post("/admin/models/create", json_data={
            "name": name,
            "description": description,
            "accuracy": accuracy,
        })

    def activate_model(self, name: str) -> Dict:
        """Activate a model for serving (admin only)."""
        return self._post(f"/admin/models/{name}/activate")

    def compare_models(
        self,
        model_names: List[str],
        name: str = "Model Comparison",
        dataset: Optional[str] = None,
    ) -> ModelComparison:
        """Run a side-by-side comparison of models."""
        data = self._post("/models/compare/run", json_data={
            "name": name,
            "model_names": model_names,
            "dataset": dataset,
        })
        return ModelComparison(
            comparison_id=data.get("id", ""),
            name=data.get("name", ""),
            results=data.get("results", {}),
            winner=data.get("winner"),
            metric_used=data.get("metric_used", "accuracy"),
        )

    def leaderboard(self, metric: str = "accuracy", limit: int = 10) -> List[Dict]:
        """Get model leaderboard ranked by a metric."""
        return self._get("/models/compare/leaderboard", params={
            "metric": metric,
            "limit": limit,
        }).get("leaderboard", [])

    # --- A/B Testing ---

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        description: str = "",
        traffic_allocation: float = 1.0,
        primary_metric: str = "conversion",
    ) -> Dict:
        """Create a new A/B experiment (admin only)."""
        return self._post("/experiments/create", json_data={
            "name": name,
            "description": description,
            "variants": variants,
            "traffic_allocation": traffic_allocation,
            "primary_metric": primary_metric,
        })

    def start_experiment(self, experiment_id: str) -> Dict:
        """Start a draft experiment (admin only)."""
        return self._post(f"/experiments/{experiment_id}/start")

    def stop_experiment(self, experiment_id: str, winner: Optional[str] = None) -> Dict:
        """Stop a running experiment (admin only)."""
        params = {}
        if winner:
            params["winner"] = winner
        return self._post(f"/experiments/{experiment_id}/stop", params=params)

    def list_experiments(self, status: Optional[str] = None) -> List[Dict]:
        """List all experiments."""
        params = {}
        if status:
            params["status"] = status
        return self._get("/experiments/list", params=params)

    def experiment_results(self, experiment_id: str) -> ExperimentResult:
        """Get detailed experiment results."""
        data = self._get(f"/experiments/{experiment_id}/results")
        return ExperimentResult(
            experiment_id=experiment_id,
            name=data.get("experiment", {}).get("name", ""),
            status=data.get("experiment", {}).get("status", ""),
            results=data.get("results", {}),
            significance=data.get("significance"),
            winner=data.get("experiment", {}).get("winner"),
        )

    def get_variant(self, experiment_id: str) -> Dict:
        """Get the assigned variant for the current user."""
        return self._get(f"/experiments/{experiment_id}/variant")

    def track_event(
        self,
        experiment_id: str,
        event_type: str = "conversion",
        event_name: str = "",
        event_value: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Track an event for an experiment."""
        return self._post("/experiments/track", json_data={
            "experiment_id": experiment_id,
            "event_type": event_type,
            "event_name": event_name,
            "event_value": event_value,
            "metadata": metadata or {},
        })

    # --- Benchmarking ---

    def benchmark_api(
        self,
        endpoint: str,
        method: str = "GET",
        iterations: int = 100,
    ) -> BenchmarkResult:
        """Benchmark an API endpoint."""
        data = self._post("/benchmarks/api", params={
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
        })
        return BenchmarkResult(
            avg_latency_ms=data.get("avg_latency_ms", 0),
            min_latency_ms=data.get("min_latency_ms", 0),
            max_latency_ms=data.get("max_latency_ms", 0),
            median_latency_ms=data.get("median_latency_ms", 0),
            p95_latency_ms=data.get("p95_latency_ms", 0),
            p99_latency_ms=data.get("p99_latency_ms", 0),
            requests_per_second=data.get("requests_per_second", 0),
            error_rate=data.get("error_rate", 0),
            raw=data,
        )

    def benchmark_model(
        self,
        model_name: str,
        iterations: int = 50,
    ) -> BenchmarkResult:
        """Benchmark a model's inference speed."""
        data = self._post(f"/benchmarks/model/{model_name}", params={
            "iterations": iterations,
        })
        return BenchmarkResult(
            avg_latency_ms=data.get("avg_latency_ms", 0),
            min_latency_ms=data.get("min_latency_ms", 0),
            max_latency_ms=data.get("max_latency_ms", 0),
            median_latency_ms=data.get("median_latency_ms", 0),
            p95_latency_ms=data.get("p95_latency_ms", 0),
            p99_latency_ms=data.get("p99_latency_ms", 0),
            requests_per_second=data.get("requests_per_second", 0),
            error_rate=data.get("error_rate", 0),
            raw=data,
        )

    def benchmark_history(self, limit: int = 20) -> List[Dict]:
        """Get benchmark history."""
        return self._get("/benchmarks/history", params={"limit": limit})

    # --- System ---

    def system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return self._get("/system/info")

    def system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        return self._get("/system/resources")

    def list_endpoints(self) -> List[Dict]:
        """List all registered API endpoints."""
        return self._get("/system/endpoints").get("endpoints", [])

    # --- Jobs ---

    def list_jobs(self) -> List[Dict]:
        """List all jobs."""
        return self._get("/jobs")

    def get_job(self, job_id: str) -> Dict:
        """Get a specific job."""
        return self._get(f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a job."""
        return self._post(f"/jobs/{job_id}/cancel")

    # --- Context Manager ---

    def __enter__(self) -> "JarvisClient":
        return self

    def __exit__(self, *args):
        self._session.close()

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
