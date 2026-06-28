# JarvisAI Deployment Guide

**Last Updated:** June 28, 2026

## Quick Start

```bash
# 1. Clone and setup
cp .env.example .env
# Edit .env and set required values

# 2. Local development with Docker Compose
docker-compose up --build

# 3. Verify
curl http://localhost:8000/health
```

## Deployment Options

### Option 1: Docker Compose (Local/Dev)

```bash
docker-compose up --build
# Access at http://localhost:8000
```

For GPU support: `docker-compose -f docker-compose.gpu.yml up --build`
For staging: `docker-compose -f docker-compose.staging.yml up --build`
For production: `docker-compose -f docker-compose.prod.yml up --build`

### Option 2: Azure Container Apps (Recommended for Production)

**Prerequisites:**
- Azure CLI (v2.50.0+): `winget install Microsoft.AzureCLI`
- Azure Developer CLI: `winget install Microsoft.Azd`
- Docker Desktop

```bash
az login
azd auth login
azd init
azd up
```

This provisions: Container Registry, Container Apps Environment, PostgreSQL, Redis, and Log Analytics.

### Option 3: Kubernetes

```bash
kubectl apply -f helm/    # Helm chart
# OR
kubectl apply -f infrastructure/kubernetes/
```

### Option 4: Fly.io

```bash
fly deploy  # uses fly.toml
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JARVIS_ENV` | Environment (development/staging/production) | development |
| `DATABASE_URL` | PostgreSQL connection string | sqlite:///./jarvis.db |
| `REDIS_URL` | Redis connection string | redis://localhost:6379/0 |
| `SECRET_KEY` | Application secret key | (required) |
| `LLM_PROVIDER` | LLM provider (openai/ollama/groq) | (optional) |
| `OPENAI_API_KEY` | OpenAI API key | (optional) |
| `OLLAMA_BASE_URL` | Ollama server URL | http://localhost:11434 |

### Scaling (Azure)

Edit `infrastructure/containerApp.bicep`:
- Min replicas: 1
- Max replicas: 10
- CPU threshold: 70%

## Database Migrations

```bash
alembic upgrade head
```

## Monitoring

- Health check: `GET /health`
- Dashboard: `GET /health/dashboard`
- Metrics: Prometheus at `:9090/metrics` (if enabled)
- Logs: `azd logs --follow` or Azure Portal → Application Insights

## Troubleshooting

**Container won't start:**
```bash
docker logs <container_id>
# Ensure app listens on 0.0.0.0:8000
```

**Database connection failed:**
```bash
# Test connectivity
docker-compose exec postgres pg_isready -U jarvis
```

**High memory usage:**
- Increase memory in `infrastructure/containerApp.bicep`
- Scale horizontally: `az containerapp update --min-replicas 2 --max-replicas 20`

## Cost Estimation (Azure)

| Environment | Monthly Estimate |
|-------------|-----------------|
| Development | ~$61-101 |
| Production | ~$395-595 |

## Security

- All secrets stored in Azure Key Vault (managed identity)
- HTTPS enforced on Container Apps
- RBAC for access control
- Never commit `.env` files — use `.env.example` as template

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `ci-cd.yml` — test, lint, coverage on PR/push to main
- `azure-dev.yml` — deploy to Azure on merge to main
- `security-audit.yml` — Bandit security scanning
- `docker-publish.yml` — build and push Docker image

## Cleanup (Azure)

```bash
azd down --force --purge
```
