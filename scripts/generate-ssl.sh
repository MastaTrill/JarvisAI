#!/bin/bash
# Generate self-signed SSL certificates for development
# For production, use Let's Encrypt or your CA

mkdir -p config/ssl

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/ssl/server.key \
  -out config/ssl/server.crt \
  -subj "/CN=localhost/O=JarvisAI/C=US" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

chmod 600 config/ssl/server.key
chmod 644 config/ssl/server.crt

echo "SSL certificates generated in config/ssl/"
echo "For production, replace with certificates from Let's Encrypt or your CA"