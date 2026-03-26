# Cloud Deployment Guide for Jarvis AI

## Docker
- Build the image: `docker build -t jarvis-ai .`
- Run locally: `docker run -p 8000:8000 jarvis-ai`

## Kubernetes
- Deploy: `kubectl apply -f k8s-deployment.yaml`
- Access: Port-forward or expose service as needed

## Cloud Providers
- Push Docker image to your registry (e.g., AWS ECR, GCP GCR, Azure ACR)
- Update `k8s-deployment.yaml` with your image path
- Deploy to your managed Kubernetes cluster

## Notes
- Ensure environment variables and secrets are managed securely
- For production, set resource limits in your deployment manifest
