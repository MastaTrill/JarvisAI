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

# TODO: Integrate with Prometheus Alertmanager for alerting
# TODO: Add graceful shutdown and probe endpoints
