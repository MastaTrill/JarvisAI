# Jarvis AI Project

A comprehensive AI/ML platform with modular FastAPI backend, versioned API endpoints, admin dashboard, audit/compliance, real-time collaboration, plugin system, and cloud deployment support.

![Coverage](https://img.shields.io/badge/coverage-36%25-yellow)
![Test Coverage](https://img.shields.io/badge/tests-245%20passed-brightgreen)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## 🚀 Features

- **Versioned API**: All endpoints accessible under `/v1` (see OpenAPI docs at `/docs`)
- **Admin Dashboard**: Manage users, models, jobs, and system settings
- **Audit & Compliance**: Log actions, export audit logs, GDPR/CCPA endpoints
- **Collaboration**: Real-time annotation and feedback, WebSocket support
- **Plugin System**: Easily extend platform with custom plugins
- **Authentication & RBAC**: OAuth2, API Key, admin role enforcement
- **Cloud Deployment**: Docker, Azure, and cross-platform scripts
- **Machine Learning**: Custom numpy-based neural networks, advanced models
- **Data Processing**: pandas and numpy for data manipulation
- **Visualization**: matplotlib and seaborn for data visualization
- **Model Tracking**: Structured logging and metrics tracking
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **LLM Integration**: OpenAI, Ollama, Groq support
- **Self-Healing**: Automated error detection and recovery
- **Multimodal AI**: Text, image, and voice processing

## 📁 Project Structure

```text
JarvisAI/
├── main_api.py                   # FastAPI entrypoint (re-exports jarvis_api.app)
├── jarvis_api.py                 # Main API module with all routers
├── agent_api.py                  # Agent tools, chat, memory endpoints
├── agent_memory.py               # Memory management
├── agent_task_memory.py          # Task persistence
├── admin_api.py                  # Admin API endpoints
├── admin_dashboard.py            # Admin dashboard endpoints
├── audit_api.py                  # Audit/compliance endpoints
├── audit_trail.py                # Audit logging
├── authentication.py             # JWT + OAuth2 authentication
├── auth_helpers.py               # Auth dependency helpers
├── cache.py                      # Memory + LLM cache
├── celery_app.py                 # Celery task queue config
├── celery_tasks.py               # Background task definitions
├── cloud_connectors.py           # S3/cloud storage connectors
├── collab_api.py                 # Collaboration endpoints
├── dashboard.py                  # Streamlit dashboard
├── database.py                   # Database engine/session
├── database_models.py            # SQLAlchemy ORM models
├── db_config.py                  # DB configuration
├── infra_api.py                  # Infrastructure endpoints
├── jobs_api.py                   # Job management endpoints
├── jobs_persistent.py            # Persistent job storage
├── llm_groq.py                   # Groq LLM integration
├── llm_integration.py            # LLM abstraction layer
├── llm_ollama.py                 # Ollama LLM integration
├── llm_openai.py                 # OpenAI LLM integration
├── ml_advanced_api.py            # Advanced ML endpoints
├── models_device_api.py          # Device-aware model serving
├── models_drift_api.py           # Drift detection endpoints
├── models_external_api.py        # External model server integration
├── models_registry.py            # In-memory model registry
├── models_user.py                # User model
├── models_versioning.py          # Model versioning (ORM)
├── models_versioning_api.py      # Model versioning endpoints
├── monitoring.py                 # System monitoring
├── observability.py              # OpenTelemetry tracing
├── plugins_api.py                # Plugin management endpoints
├── security_api.py               # Security/RBAC endpoints
├── advanced_features/            # Advanced AI modules
│   ├── ai_workflow_automation.py
│   ├── explainable_ai.py
│   ├── federated_learning.py
│   ├── knowledge_integration.py
│   ├── live_data_viz.py
│   ├── multimodal_ai.py
│   ├── nlu_advanced.py
│   ├── orchestrator.py
│   ├── quantum_optimization.py
│   └── self_healing.py
├── plugins/                      # Plugin system
├── src/                          # ML models and data processing
│   ├── data/                     # Data processors
│   ├── models/                   # Neural network implementations
│   ├── training/                 # Training pipelines
│   └── inference/                # Inference/prediction
├── tests/                        # Unit/integration tests (245 tests)
├── Dockerfile                    # Production Docker image
├── docker-compose.yml            # Full stack orchestration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MastaTrill/JarvisAI.git
   cd JarvisAI
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## 🚦 Quick Start

### Run the API server:

```bash
python main_api.py
# or
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker Compose (recommended):

```bash
docker compose up --build -d
```

This starts: `jarvis-api` (port 8000), `jarvis-worker` (Celery), `ollama` (port 11434), `postgres` (port 5432), `redis` (port 6379).

### Access the platform:

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Admin Dashboard**: http://localhost:8000/admin

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_auth.py -v
```

**Current status**: 245 passed, 1 skipped

## 🔒 Authentication & RBAC

- OAuth2, API Key, and admin dashboard authentication
- Role-based access control (admin endpoints require admin role)
- JWT tokens with configurable expiry

## 📝 Example API Requests

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Register User:**

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username":"newuser","password":"secret","email":"new@user.com"}'
```

**Login:**

```bash
curl -X POST http://localhost:8000/token \
  -d "username=newuser&password=secret"
```

## ☁️ Deployment

- **Docker:** `docker build -t jarvisai . && docker run -p 8000:8000 jarvisai`
- **Docker Compose:** `docker compose up --build -d`
- **Azure:** See `DEPLOYMENT.md` and `azure.yaml`
- **Kubernetes:** See `k8s-deployment.yaml` and `infrastructure/kubernetes/`
- **GitHub Actions:** CI/CD workflows in `.github/workflows/`

## 🔒 Security

- Audit logging for all sensitive admin actions
- Role-based access control (RBAC) for users and admins
- Rate limiting, secure headers, and CORS enabled
- GDPR/CCPA compliance endpoints
- See `SECURITY.md` for more details

## 🧩 Plugins

- Add custom plugins in `plugins/` and register with the main API
- See `REGISTRY.md` for community plugins

## 📚 Documentation

- OpenAPI docs: `/docs`
- Redoc: `/redoc`
- See `docs/` for tutorials and architecture guides
- See `DEPLOYMENT.md` for deployment instructions

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

See `CONTRIBUTING.md` for details.

## 📞 Support

For issues or questions:

1. Check the troubleshooting section in `docs/troubleshooting.md`
2. Review the test files for usage examples
3. Create an issue at https://github.com/MastaTrill/JarvisAI/issues

---

**Last Updated:** May 16, 2026
