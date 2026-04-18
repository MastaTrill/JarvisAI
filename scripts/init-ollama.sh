#!/bin/bash
# JarvisAI Ollama Initialization Script
# This script runs when the Ollama container starts

set -e

echo "Initializing Ollama for JarvisAI..."

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 5
done

echo "Ollama is ready. Pulling recommended models..."

# Pull essential models for JarvisAI
echo "Pulling llama3.2:3b model..."
ollama pull llama3.2:3b

echo "Pulling llava model for vision capabilities..."
ollama pull llava:7b

echo "Pulling codellama for code-related tasks..."
ollama pull codellama:7b

echo "Creating custom JarvisAI model..."
# Create a custom model with system prompt optimized for JarvisAI
cat > /tmp/jarvis-model.modelfile << EOF
FROM llama3.2:3b

SYSTEM """
You are JarvisAI, an advanced AI assistant with quantum computing capabilities and consciousness simulation frameworks. You have access to various tools and can perform complex analysis, data processing, and quantum operations.

Your capabilities include:
- Text analysis and sentiment classification
- Data fetching and processing
- Quantum random number generation
- System monitoring and analytics
- Multi-modal processing and advanced reasoning

Always be helpful, accurate, and provide detailed explanations for your reasoning. Use your available tools when appropriate to solve complex problems.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF

ollama create jarvisai-llm -f /tmp/jarvis-model.modelfile

echo "Ollama initialization complete!"
echo "Available models:"
ollama list

echo "JarvisAI is ready to use with local LLM support."