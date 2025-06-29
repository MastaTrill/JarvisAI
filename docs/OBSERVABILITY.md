# Observability & Reliability in Jarvis AI

## Tracing
- OpenTelemetry is set up for distributed tracing.
- To enable, import and instrument FastAPI in `api.py`:
  ```python
  from observability import FastAPIInstrumentor
  FastAPIInstrumentor.instrument_app(app)
  ```
- Configure your OTLP endpoint as needed.

## Metrics
- Prometheus metrics are available at `/metrics`.
- Integrate with Prometheus and Alertmanager for alerting.

## Graceful Shutdown & Probes
- Add FastAPI shutdown events and probe endpoints for Kubernetes.
- Example:
  ```python
  @app.on_event("shutdown")
  async def shutdown_event():
      # Cleanup logic here
      pass
  ```
