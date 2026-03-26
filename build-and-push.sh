#!/bin/bash
# Automated build and publish for JarvisAI Docker image
# Usage: ./build-and-push.sh

set -e
IMAGE=ghcr.io/mastatrill/jarvisai:latest

# Build Docker image

echo "[JarvisAI] Building Docker image..."
if docker buildx version >/dev/null 2>&1; then
	docker buildx build --push -t $IMAGE .
else
	docker build -t $IMAGE .
	docker push $IMAGE
fi

echo "[JarvisAI] Image published: $IMAGE"