#!/bin/bash
# JarvisAI Production Deployment Script
# Usage: ./scripts/deploy.sh

set -e

echo "=== JarvisAI Production Deployment ==="
echo "Starting deployment..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Start Docker Desktop and try again"
    exit 1
fi

# Create necessary directories
mkdir -p logs data models backups

# Pull latest images
echo "Pulling Docker images..."
docker compose -f docker-compose.prod.yml pull

# Stop existing containers
echo "Stopping existing containers..."
docker compose -f docker-compose.prod.yml down

# Start services
echo "Starting services..."
docker compose -f docker-compose.prod.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 10

# Run health checks
echo "Running health checks..."
docker compose -f docker-compose.prod.yml exec -T api curl -f http://localhost:8000/health || exit 1

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Access points:"
echo "  - API: http://localhost:8080"
echo "  - Docs: http://localhost:8080/docs"
echo "  - Admin: http://localhost:8080/admin"
echo "  - Grafana: http://localhost:3001"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "View logs: docker compose -f docker-compose.prod.yml logs -f"