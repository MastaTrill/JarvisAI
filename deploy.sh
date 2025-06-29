#!/bin/bash
# One-click deploy script for Jarvis AI Platform (Docker Compose)
# Usage: ./deploy.sh

set -e

# Build the Docker image
echo "[Jarvis] Building Docker image..."
docker build -t jarvis-ai .

echo "[Jarvis] Starting container..."
docker run -d --name jarvis-ai -p 8000:8000 --env-file .env jarvis-ai

echo "[Jarvis] Deployment complete!"
echo "Visit: http://localhost:8000/"
