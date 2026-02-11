@echo off
REM JarvisAI Staging Deployment Script for Windows
REM This script deploys the JarvisAI platform to staging environment

echo 🚀 Starting JarvisAI Staging Deployment
echo =======================================

REM Create necessary directories
echo Creating necessary directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist logs mkdir logs
if not exist monitoring\grafana\provisioning\datasources mkdir monitoring\grafana\provisioning\datasources
if not exist monitoring\grafana\provisioning\dashboards mkdir monitoring\grafana\provisioning\dashboards

REM Set environment variables
set JARVIS_ENV=staging
set SECRET_KEY=staging-secret-key-change-in-production

REM Pull latest images
echo Pulling latest Docker images...
docker-compose -f docker-compose.staging.yml pull

REM Build custom images
echo Building custom Docker images...
docker-compose -f docker-compose.staging.yml build --no-cache

REM Start services
echo Starting JarvisAI staging services...
docker-compose -f docker-compose.staging.yml up -d

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 30 /nobreak > nul

REM Check service health
echo Checking service health...

REM Check API health
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API service is healthy
) else (
    echo ❌ API service is not responding
)

echo.
echo 🎉 JarvisAI Staging Deployment Complete!
echo ==========================================
echo.
echo Service URLs:
echo   🌐 API:         http://localhost:8000
echo   📊 API Docs:    http://localhost:8000/api/v3/docs
echo   📈 Grafana:     http://localhost:3000 (admin/admin)
echo   📋 Prometheus:  http://localhost:9090
echo   🗄️  Database:    localhost:5432
echo   🗝️  Redis:       localhost:6379
echo.
echo Default Admin Credentials:
echo   Username: admin
echo   Password: admin123
echo   Email:    admin@staging.jarvis.ai
echo.
echo ⚠️  IMPORTANT: Change default passwords before production deployment!
echo.
echo To view logs: docker-compose -f docker-compose.staging.yml logs -f
echo To stop:      docker-compose -f docker-compose.staging.yml down
echo To restart:   docker-compose -f docker-compose.staging.yml restart

pause