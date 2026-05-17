# Self-signed SSL certificate for development
# Run: openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl.key -out ssl.crt -subj "/CN=localhost"
# Production: Use proper certificates from Let's Encrypt or your CA