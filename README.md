# Jarvis AI Project

A comprehensive AI/ML platform with modular FastAPI backend, versioned API endpoints, admin dashboard, audit/compliance, real-time collaboration, plugin system, and cloud deployment support.

![Build Status](https://github.com/MastaTrill/JarvisAI/actions/workflows/ci-cd.yml/badge.svg)

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

## Project Structure

```text
JarvisAI/
├── main_api.py                   # FastAPI entrypoint
├── jarvis_api.py                 # Main API module with all routers
├── src/                          # Organized source packages
│   ├── api/                      # API route modules (admin, agent, audit, etc.)
│   ├── ai/                       # LLM integrations, RAG, code sandbox
│   ├── agents/                   # Agent memory, config, orchestration
│   ├── infra/                    # Database, cache, auth, celery
│   ├── ml/                       # Model registry, versioning, analytics
│   └── core/                     # Core utilities
├── tests/                        # Unit/integration tests
├── advanced_features/            # Advanced AI modules
├── plugins/                      # Plugin system
├── infrastructure/               # IaC (Bicep, Terraform, K8s)
├── helm/                         # Helm chart
├── Dockerfile                    # Production Docker image
├── docker-compose.yml            # Full stack orchestration
├── requirements.txt              # Python dependencies
└── DEPLOYMENT.md                 # Deployment guide
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

**Current status**: 315 passed, 2 skipped, 0 failures

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
