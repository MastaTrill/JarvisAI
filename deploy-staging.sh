#!/bin/bash

# JarvisAI Staging Deployment Script
# This script deploys the JarvisAI platform to staging environment

set -e

echo "🚀 Starting JarvisAI Staging Deployment"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data models logs monitoring/grafana/provisioning/datasources monitoring/grafana/provisioning/dashboards

# Set environment variables
export JARVIS_ENV=staging
export SECRET_KEY=${SECRET_KEY:-"staging-secret-key-change-in-production"}

# Pull latest images
print_status "Pulling latest Docker images..."
docker-compose -f docker-compose.staging.yml pull

# Build custom images
print_status "Building custom Docker images..."
docker-compose -f docker-compose.staging.yml build --no-cache

# Start services
print_status "Starting JarvisAI staging services..."
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check API health
if curl -f http://localhost:8000/health &> /dev/null; then
    print_success "API service is healthy"
else
    print_error "API service is not responding"
fi

# Check database health
if docker-compose -f docker-compose.staging.yml exec -T jarvis-db pg_isready -U jarvis &> /dev/null; then
    print_success "Database service is healthy"
else
    print_warning "Database service health check failed"
fi

# Check Redis health
if docker-compose -f docker-compose.staging.yml exec -T jarvis-redis redis-cli ping | grep -q PONG; then
    print_success "Redis service is healthy"
else
    print_warning "Redis service health check failed"
fi

# Run database migrations/initialization
print_status "Running database initialization..."
docker-compose -f docker-compose.staging.yml exec -T jarvis-api python -c "
import sys
sys.path.append('.')
from database import init_db
init_db()
print('Database initialized successfully')
" || print_warning "Database initialization may have issues"

# Display service URLs
echo ""
echo "🎉 JarvisAI Staging Deployment Complete!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  🌐 API:         http://localhost:8000"
echo "  📊 API Docs:    http://localhost:8000/api/v3/docs"
echo "  📈 Grafana:     http://localhost:3000 (admin/admin)"
echo "  📋 Prometheus:  http://localhost:9090"
echo "  🗄️  Database:    localhost:5432"
echo "  🗝️  Redis:       localhost:6379"
echo ""
echo "Default Admin Credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo "  Email:    admin@staging.jarvis.ai"
echo ""
print_warning "⚠️  IMPORTANT: Change default passwords before production deployment!"
echo ""
echo "To view logs: docker-compose -f docker-compose.staging.yml logs -f"
echo "To stop:      docker-compose -f docker-compose.staging.yml down"
echo "To restart:   docker-compose -f docker-compose.staging.yml restart"