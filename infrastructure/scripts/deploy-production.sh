#!/bin/bash
# Jarvis AI Production Deployment Script
# Deploys all Kubernetes resources in the correct order

set -e

NAMESPACE="jarvis-production"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../kubernetes/production"

echo "🚀 Deploying Jarvis AI to Production"
echo "====================================="

# Check kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster. Check your kubeconfig."
    exit 1
fi

echo "📦 Creating namespace..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

echo "🔐 Creating secrets (ensure you've updated with real values)..."
kubectl apply -f "$K8S_DIR/secrets.yaml"

echo "⚙️ Creating ConfigMap..."
kubectl apply -f "$K8S_DIR/configmap.yaml"

echo "👤 Creating RBAC resources..."
kubectl apply -f "$K8S_DIR/rbac.yaml"

echo "💾 Creating Persistent Volume Claims..."
kubectl apply -f "$K8S_DIR/pvc.yaml"

echo "🚀 Deploying applications..."
kubectl apply -f "$K8S_DIR/deployment.yaml"

echo "🌐 Creating services..."
kubectl apply -f "$K8S_DIR/service.yaml"

echo "🔀 Creating ingress..."
kubectl apply -f "$K8S_DIR/ingress.yaml"

echo "📈 Creating HPA..."
kubectl apply -f "$K8S_DIR/hpa.yaml"

echo "🛡️ Creating Pod Disruption Budgets..."
kubectl apply -f "$K8S_DIR/pod-disruption-budget.yaml"

echo ""
echo "⏳ Waiting for deployments to be ready..."
kubectl rollout status deployment/jarvis-api -n "$NAMESPACE" --timeout=300s
kubectl rollout status deployment/jarvis-worker -n "$NAMESPACE" --timeout=300s

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Current status:"
kubectl get pods -n "$NAMESPACE"
echo ""
kubectl get svc -n "$NAMESPACE"
echo ""
kubectl get ingress -n "$NAMESPACE"
