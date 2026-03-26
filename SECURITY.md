# Security Hardening for Jarvis AI

## Dependency Scanning
- Use `bandit` for Python security checks: `bandit -r .`
- Enable Dependabot in your repository for automated alerts.

## Current Advisory Note
- As of March 26, 2026, the repo dependency cleanup reduced the actionable dependency advisories to one remaining upstream alert affecting `Pygments 2.19.2`.
- The remaining `Pygments` advisory currently has no published fixed release in upstream vulnerability metadata, so there is no clean version bump to apply yet.
- Treat this as a watch item: update `Pygments` promptly once an upstream patched release is published and Dependabot or your lockfile tooling can resolve to it.

## HTTPS & Secure Headers
- Deploy behind a reverse proxy (e.g., Nginx, Traefik) with HTTPS.
- Use FastAPI middleware for secure headers (see below).

## Secrets Management
- Store secrets in environment variables or a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault).
- Never commit real secrets to version control.

## Example: Secure Headers Middleware
```python
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "localhost"])
```
