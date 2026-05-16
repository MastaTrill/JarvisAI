"""
Observability & Tracing Setup for Jarvis AI
- OpenTelemetry tracing for FastAPI
- Prometheus metrics integration
- Placeholder for Alertmanager integration
"""
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace

# Initialize tracing
provider = TracerProvider()
trace.set_tracer_provider(provider)
span_processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(span_processor)

# Usage in api.py:
# from observability import FastAPIInstrumentor
# FastAPIInstrumentor.instrument_app(app)

# Prometheus metrics are exposed at /metrics (already in api.py)

# Prometheus Alertmanager integration
import requests

def send_alert_to_alertmanager(alertmanager_url, alert_name, description, severity="warning", labels=None):
	"""
	Send an alert to Prometheus Alertmanager.
	:param alertmanager_url: Base URL of Alertmanager (e.g., http://localhost:9093/api/v1/alerts)
	:param alert_name: Name of the alert
	:param description: Description of the alert
	:param severity: Alert severity (default: warning)
	:param labels: Additional labels as dict
	"""
	payload = [{
		"labels": {
			"alertname": alert_name,
			"severity": severity,
			**(labels or {})
		},
		"annotations": {
			"description": description
		}
	}]
	try:
		response = requests.post(alertmanager_url, json=payload, timeout=5)
		response.raise_for_status()
		return True
	except Exception as e:
		print(f"Failed to send alert: {e}")
		return False

# Probe endpoints and graceful shutdown
from fastapi import APIRouter, Response
import threading

probe_router = APIRouter()

@probe_router.get("/healthz")
def liveness_probe():
	return {"status": "ok"}

@probe_router.get("/readyz")
def readiness_probe():
	# Add custom readiness checks as needed
	return {"status": "ready"}

shutdown_event = threading.Event()

def register_graceful_shutdown(app):
	@app.on_event("shutdown")
	async def shutdown_handler():
		print("Graceful shutdown initiated.")
		shutdown_event.set()
