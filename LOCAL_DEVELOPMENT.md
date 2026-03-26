# JarvisAI Local Development Guide

Quick guide for testing JarvisAI locally before Azure deployment.

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Start all services (PostgreSQL, Redis, API)
docker-compose -f docker-compose.local.yml up -d

# 2. View logs
docker-compose -f docker-compose.local.yml logs -f api

# 3. Test the API
curl http://localhost:8000/health

# 4. Open API documentation
# Visit: http://localhost:8000/docs

# 5. Stop services
docker-compose -f docker-compose.local.yml down
```

### Using Python Virtual Environment

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL and Redis (with Docker)
docker run -d --name postgres-local \
  -e POSTGRES_DB=jarvisai \
  -e POSTGRES_USER=jarvisadmin \
  -e POSTGRES_PASSWORD=localdevpassword \
  -p 5432:5432 \
  postgres:16

docker run -d --name redis-local \
  -p 6379:6379 \
  redis:7-alpine redis-server --requirepass localdevpassword

# 4. Set environment variables
# Copy .env.example to .env and update:
cp .env.example .env

# Edit .env:
DATABASE_URL=postgresql://jarvisadmin:localdevpassword@localhost:5432/jarvisai
REDIS_URL=redis://:localdevpassword@localhost:6379/0
SECRET_KEY=local-dev-secret-key

# 5. Run the API
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## 🧪 Testing Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/models
```

### Interactive API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🛠️ Common Tasks

### View Container Logs

```bash
# API logs
docker-compose -f docker-compose.local.yml logs -f api

# Database logs
docker-compose -f docker-compose.local.yml logs -f postgres

# Redis logs
docker-compose -f docker-compose.local.yml logs -f redis
```

### Access Database

```bash
# Using psql
docker exec -it jarvisai-postgres psql -U jarvisadmin -d jarvisai

# List tables
\dt

# Query
SELECT * FROM users;
```

### Access Redis

```bash
# Using redis-cli
docker exec -it jarvisai-redis redis-cli -a localdevpassword

# Test connection
PING

# List keys
KEYS *
```

### Rebuild Container

```bash
# Rebuild after code changes
docker-compose -f docker-compose.local.yml build api

# Restart service
docker-compose -f docker-compose.local.yml up -d api
```

## 🔍 Debugging

### Check API Container Status

```bash
docker-compose -f docker-compose.local.yml ps
```

### Access API Container

```bash
docker exec -it jarvisai-api bash
```

### View Full Logs

```bash
docker-compose -f docker-compose.local.yml logs --tail=100
```

## 🧹 Cleanup

### Stop All Services

```bash
docker-compose -f docker-compose.local.yml down
```

### Remove Volumes (Delete Data)

```bash
docker-compose -f docker-compose.local.yml down -v
```

### Remove All Containers and Images

```bash
docker-compose -f docker-compose.local.yml down --rmi all -v
```

## 📝 Environment Variables

Key environment variables for local development (set in `.env`):

```env
# Database
DATABASE_URL=postgresql://jarvisadmin:localdevpassword@localhost:5432/jarvisai

# Redis
REDIS_URL=redis://:localdevpassword@localhost:6379/0

# Application
SECRET_KEY=local-dev-secret-key-change-me
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=DEBUG

# API
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=2

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

## 🔄 Hot Reload

The Docker Compose setup includes volume mounts for hot reloading:

- Changes to `api.py` trigger automatic reload
- No need to rebuild container for code changes
- Great for rapid development

## 🚢 Prepare for Azure Deployment

Before deploying to Azure, test in production mode:

```bash
# 1. Build production image
docker build -t jarvisai:prod .

# 2. Run with production settings
docker run -p 8000:8000 \
  -e DATABASE_URL=your-real-db-url \
  -e REDIS_URL=your-real-redis-url \
  -e SECRET_KEY=your-secure-secret \
  -e ENVIRONMENT=production \
  jarvisai:prod

# 3. Test thoroughly
curl http://localhost:8000/health
```

## 📚 Next Steps

Once local testing is complete:

1. Review [DEPLOYMENT.md](DEPLOYMENT.md) for Azure deployment
2. Run `azd up` to deploy to Azure
3. Monitor deployment at https://portal.azure.com

---

**For Azure Deployment:** See [DEPLOYMENT.md](DEPLOYMENT.md)
**For Issues:** Check container logs and health endpoints
