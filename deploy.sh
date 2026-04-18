#!/bin/bash
# JarvisAI Production Deployment Script
# This script sets up and deploys JarvisAI in a production environment

set -e

echo "🚀 JarvisAI Production Deployment"
echo "================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env.prod exists
if [ ! -f .env.prod ]; then
    echo "⚠️  .env.prod file not found. Creating from template..."
    cp .env.prod.example .env.prod
    echo "📝 Please edit .env.prod with your actual configuration values before continuing."
    echo "   Important: Change SECRET_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD, etc."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data models config/ssl
chmod +x scripts/init-ollama.sh

# Load environment variables
set -a
source .env.prod
set +a

echo "🐳 Starting JarvisAI production environment..."

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down || true

# Start the services
echo "🏗️  Building and starting services..."
docker-compose -f docker-compose.prod.yml up -d --build

echo "⏳ Waiting for services to be healthy..."

# Wait for PostgreSQL
echo "📊 Waiting for PostgreSQL..."
timeout=60
while ! docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U jarvisadmin -d jarvisai_prod > /dev/null 2>&1; do
    if [ $timeout -le 0 ]; then
        echo "❌ PostgreSQL failed to start"
        exit 1
    fi
    echo "Waiting for PostgreSQL... ($timeout seconds remaining)"
    sleep 5
    timeout=$((timeout - 5))
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "🔄 Waiting for Redis..."
timeout=30
while ! docker-compose -f docker-compose.prod.yml exec -T redis redis-cli -a "$REDIS_PASSWORD" ping | grep -q PONG; do
    if [ $timeout -le 0 ]; then
        echo "❌ Redis failed to start"
        exit 1
    fi
    echo "Waiting for Redis... ($timeout seconds remaining)"
    sleep 5
    timeout=$((timeout - 5))
done
echo "✅ Redis is ready"

# Wait for API
echo "🤖 Waiting for JarvisAI API..."
timeout=60
while ! curl -f http://localhost:${API_PORT:-8080}/health > /dev/null 2>&1; do
    if [ $timeout -le 0 ]; then
        echo "❌ JarvisAI API failed to start"
        exit 1
    fi
    echo "Waiting for API... ($timeout seconds remaining)"
    sleep 5
    timeout=$((timeout - 5))
done
echo "✅ JarvisAI API is ready"

# Wait for Ollama (optional)
echo "🧠 Checking Ollama status..."
if curl -f http://localhost:${OLLAMA_PORT:-11435}/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is ready"
else
    echo "⚠️  Ollama is not ready yet (this is normal, models are downloading)"
fi

echo ""
echo "🎉 JarvisAI Production Deployment Complete!"
echo "=========================================="
echo ""
echo "📊 Services Status:"
echo "   🌐 JarvisAI API:    http://localhost:${API_PORT:-8080}"
echo "   📖 API Docs:        http://localhost:${API_PORT:-8080}/docs"
echo "   📈 Prometheus:      http://localhost:${PROMETHEUS_PORT:-9090}"
echo "   📊 Grafana:         http://localhost:${GRAFANA_PORT:-3001}"
echo "   🧠 Ollama:          http://localhost:${OLLAMA_PORT:-11435}"
echo ""
echo "🔐 Default Credentials:"
echo "   Grafana Admin: admin / ${GRAFANA_PASSWORD:-admin}"
echo ""
echo "📋 Next Steps:"
echo "   1. Configure SSL certificates in config/ssl/"
echo "   2. Set up domain names and DNS"
echo "   3. Configure backup strategies"
echo "   4. Set up monitoring alerts"
echo ""
echo "📝 Useful Commands:"
echo "   View logs:     docker-compose -f docker-compose.prod.yml logs -f"
echo "   Stop services: docker-compose -f docker-compose.prod.yml down"
echo "   Restart:       docker-compose -f docker-compose.prod.yml restart"
echo ""
echo "🚀 JarvisAI is now running in production mode!"
