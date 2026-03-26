#!/bin/bash

# JarvisAI Security Audit Script
# Performs security checks on the staging deployment

set -e

echo "🔒 JarvisAI Security Audit"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if services are running
print_status "Checking if staging services are running..."
if ! docker-compose -f docker-compose.staging.yml ps | grep -q "Up"; then
    print_error "Staging services are not running. Please run deploy-staging.sh first."
    exit 1
fi

# Security checks
echo ""
print_status "Running security checks..."

# 1. Check for default passwords
print_status "Checking for default passwords..."
if docker-compose -f docker-compose.staging.yml exec -T jarvis-db psql -U jarvis -d jarvis_staging -c "SELECT username FROM users WHERE password_hash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPjYQmHqU3jPe';" | grep -q admin; then
    print_error "❌ Default admin password is still in use!"
else
    print_success "✅ Default passwords have been changed"
fi

# 2. Check SSL/TLS configuration
print_status "Checking SSL/TLS configuration..."
if curl -I http://localhost:8000/api/v3/docs 2>/dev/null | grep -q "Strict-Transport-Security"; then
    print_success "✅ HTTPS redirect configured"
else
    print_warning "⚠️  HTTPS not enforced (expected in staging)"
fi

# 3. Check security headers
print_status "Checking security headers..."
HEADERS=$(curl -I http://localhost:8000/api/v3/docs 2>/dev/null)
if echo "$HEADERS" | grep -q "X-Frame-Options"; then
    print_success "✅ X-Frame-Options header present"
else
    print_error "❌ X-Frame-Options header missing"
fi

if echo "$HEADERS" | grep -q "X-Content-Type-Options"; then
    print_success "✅ X-Content-Type-Options header present"
else
    print_error "❌ X-Content-Type-Options header missing"
fi

# 4. Check API authentication
print_status "Checking API authentication..."
if curl -s http://localhost:8000/api/v3/docs | grep -q "Authorize"; then
    print_success "✅ API authentication configured"
else
    print_warning "⚠️  API authentication may not be properly configured"
fi

# 5. Check database security
print_status "Checking database security..."
if docker-compose -f docker-compose.staging.yml exec -T jarvis-db psql -U jarvis -d jarvis_staging -c "SELECT usename FROM pg_user WHERE usename = 'jarvis';" | grep -q jarvis; then
    print_success "✅ Database user exists"
else
    print_error "❌ Database user not found"
fi

# 6. Check file permissions
print_status "Checking file permissions..."
if docker-compose -f docker-compose.staging.yml exec jarvis-api ls -la /app | grep -q "jarvisuser jarvisuser"; then
    print_success "✅ Non-root user configured"
else
    print_error "❌ Root user detected in container"
fi

# 7. Check for exposed secrets
print_status "Checking for exposed secrets..."
if grep -r "password\|secret\|key" docker-compose.staging.yml | grep -v "PASSWORD\|SECRET\|KEY" | grep -q .; then
    print_warning "⚠️  Potential secrets found in compose file"
else
    print_success "✅ No obvious secrets in compose file"
fi

# 8. Check network security
print_status "Checking network security..."
EXPOSED_PORTS=$(docker-compose -f docker-compose.staging.yml ps | grep -E ":[0-9]+->" | wc -l)
if [ "$EXPOSED_PORTS" -gt 0 ]; then
    print_warning "⚠️  $EXPOSED_PORTS ports exposed (expected in staging)"
fi

# 9. Check monitoring
print_status "Checking monitoring setup..."
if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus"; then
    print_success "✅ Prometheus monitoring active"
else
    print_warning "⚠️  Prometheus not responding"
fi

if curl -s http://localhost:3000/api/health | grep -q "ok"; then
    print_success "✅ Grafana monitoring active"
else
    print_warning "⚠️  Grafana not responding"
fi

echo ""
echo "🔒 Security Audit Complete"
echo "=========================="
echo ""
print_status "Recommendations for production:"
echo "  1. Change all default passwords"
echo "  2. Enable HTTPS with valid SSL certificates"
echo "  3. Configure proper firewall rules"
echo "  4. Set up log aggregation and monitoring"
echo "  5. Implement regular security updates"
echo "  6. Configure backup and disaster recovery"
echo "  7. Set up intrusion detection systems"
echo "  8. Implement rate limiting and DDoS protection"
echo ""
print_success "Staging environment security audit completed!"