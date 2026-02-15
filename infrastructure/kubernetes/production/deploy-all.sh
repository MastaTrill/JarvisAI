# JarvisAI Kubernetes Production Deployment Script
# This script applies all manifests in the correct order and monitors readiness.
# Usage: bash deploy-all.sh

set -e
NAMESPACE=jarvis-production
MANIFEST_DIR="$(dirname "$0")"

# 1. Create namespace
kubectl apply -f "$MANIFEST_DIR/namespace.yaml"

# 2. Apply secrets and config
kubectl apply -f "$MANIFEST_DIR/secrets.yaml"
kubectl apply -f "$MANIFEST_DIR/configmap.yaml"

# 3. RBAC
kubectl apply -f "$MANIFEST_DIR/rbac.yaml"

# 4. Persistent Volumes
kubectl apply -f "$MANIFEST_DIR/pvc.yaml"

# 5. Deployments
kubectl apply -f "$MANIFEST_DIR/deployment.yaml"

# 6. Services
kubectl apply -f "$MANIFEST_DIR/service.yaml"

# 7. Autoscaling
kubectl apply -f "$MANIFEST_DIR/hpa.yaml"

# 8. Pod Disruption Budgets
kubectl apply -f "$MANIFEST_DIR/pod-disruption-budget.yaml"

# 9. Ingress
kubectl apply -f "$MANIFEST_DIR/ingress.yaml"

# Monitor pods and services
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE
kubectl get ingress -n $NAMESPACE
