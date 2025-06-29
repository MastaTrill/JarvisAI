# docs/architecture_diagram.md

## Jarvis AI Platform - Architecture Overview

```mermaid
graph TD
    A[User/API Client] -->|REST/WebSocket| B[FastAPI API Layer]
    B --> C[Authentication & RBAC]
    B --> D[Model Registry & Versioning]
    B --> E[Job Management]
    B --> F[Plugin/Extension Registry]
    B --> G[Observability & Audit]
    D --> H[Persistent DB (Postgres/SQLite)]
    E --> H
    F --> H
    G --> H
    B --> I[Model Serving (CPU/GPU/External)]
    I --> J[Inference Engine]
    I --> K[External Model Server]
    B --> L[Admin Dashboard/UI]
    B --> M[No-Code Workflow Editor]
    B --> N[Cloud Connectors (S3, Slack, etc.)]
```

- **API Layer**: FastAPI, versioned endpoints, CORS, secure headers
- **Auth**: JWT, OAuth2, RBAC, API keys
- **Registry**: SQLAlchemy models for users, models, jobs, plugins
- **Observability**: OpenTelemetry, Prometheus, audit trail
- **Serving**: On-device, GPU, or external server (Triton, TorchServe)
- **UI**: Admin dashboard, workflow editor, plugin registry
- **Cloud**: S3, Slack, GCP, Azure connectors

---

For more, see `docs/` and `/docs/OBSERVABILITY.md`.
