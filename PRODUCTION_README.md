# JarvisAI Production Deployment Guide

This guide covers deploying JarvisAI in a production environment using Docker Compose.

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/jarvisai.git
   cd jarvisai
   ```

2. **Configure environment:**
   ```bash
   cp .env.prod.example .env.prod
   # Edit .env.prod with your actual configuration
   ```

3. **Deploy:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

## 📋 Prerequisites

- Docker and Docker Compose
- At least 8GB RAM available
- 20GB free disk space
- Linux/Windows/Mac with Docker support

## ⚙️ Configuration

### Environment Variables

Copy `.env.prod.example` to `.env.prod` and configure:

```bash
# Application
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key

# Database
POSTGRES_PASSWORD=your-postgres-password
DATABASE_URL=postgresql://jarvisadmin:${POSTGRES_PASSWORD}@postgres:5432/jarvisai_prod

# Redis
REDIS_PASSWORD=your-redis-password
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0

# External Services
OPENAI_API_KEY=your-openai-api-key
OLLAMA_BASE_URL=http://ollama:11434

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_ENABLED=True
```

### Ports Configuration

Default ports (configurable in `.env.prod`):
- JarvisAI API: 8080
- API Documentation: 8080/docs
- Prometheus: 9090
- Grafana: 3001
- Ollama: 11435
- PostgreSQL: 5433
- Redis: 6380

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   JarvisAI API  │
│     (Port 80)   │◄──►│    (Port 8000)  │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │
│   (Port 5432)   │    │   (Port 6379)   │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │    Grafana      │
│   (Port 9090)   │    │   (Port 3000)   │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐
│     Ollama      │
│  (Port 11434)   │
└─────────────────┘
```

## 📊 Services Overview

### Core Services
- **JarvisAI API**: Main application server with FastAPI
- **PostgreSQL**: Primary database for application data
- **Redis**: Cache and session storage

### AI Services
- **Ollama**: Local LLM inference with multiple models
- **OpenAI Integration**: External LLM API support

### Monitoring & Observability
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard and visualization
- **Nginx**: Reverse proxy and load balancing

## 🔧 Management Commands

### Starting Services
```bash
# Production deployment
./deploy.sh

# Manual Docker Compose
docker-compose -f docker-compose.prod.yml up -d --build
```

### Monitoring Services
```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs -f api

# Check service status
docker-compose -f docker-compose.prod.yml ps
```

### Managing Services
```bash
# Stop all services
docker-compose -f docker-compose.prod.yml down

# Restart specific service
docker-compose -f docker-compose.prod.yml restart api

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Update services
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

## 🔒 Security Considerations

### SSL/TLS Configuration
1. Obtain SSL certificates (Let's Encrypt recommended)
2. Place certificates in `config/ssl/`
3. Update Nginx configuration for SSL
4. Redirect HTTP to HTTPS

### Secrets Management
- Use Docker secrets or external secret managers
- Rotate passwords regularly
- Never commit secrets to version control

### Network Security
- Configure firewall rules
- Use internal Docker networks
- Limit exposed ports
- Enable Nginx security headers

## 📈 Monitoring & Alerting

### Accessing Dashboards
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Metrics**: http://localhost:8080/metrics

### Setting Up Alerts
1. Configure alert rules in Prometheus
2. Set up notification channels (email, Slack, etc.)
3. Create Grafana alert panels

## 🔄 Backup & Recovery

### Database Backup
```bash
# Backup PostgreSQL
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U jarvisadmin jarvisai_prod > backup.sql

# Backup Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD --rdb backup.rdb
```

### Volume Backup
```bash
# Backup Docker volumes
docker run --rm -v jarvisai_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

## 🐛 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker system
docker system df
docker system prune -a

# Check logs
docker-compose -f docker-compose.prod.yml logs
```

**Database connection issues:**
```bash
# Check PostgreSQL
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U jarvisadmin

# Reset database
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d postgres
```

**Memory issues:**
```bash
# Increase Docker memory limit
# Or reduce service resource limits in docker-compose.prod.yml
```

### Logs and Debugging
```bash
# All service logs
docker-compose -f docker-compose.prod.yml logs -f

# API specific logs
docker-compose -f docker-compose.prod.yml logs -f api

# System resource usage
docker stats
```

## 🚀 Scaling & Performance

### Horizontal Scaling
```bash
# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Add load balancer
# Configure Nginx upstream with multiple API instances
```

### Performance Tuning
- Adjust Gunicorn workers based on CPU cores
- Configure PostgreSQL memory settings
- Tune Redis memory limits
- Optimize Docker resource limits

## 📚 API Documentation

Once deployed, access:
- **API Documentation**: http://localhost:8080/docs
- **OpenAPI Spec**: http://localhost:8080/openapi.json
- **Health Check**: http://localhost:8080/health

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker and service logs
3. Check system resource usage
4. Consult the JarvisAI documentation

---

**🎉 Happy deploying with JarvisAI!**