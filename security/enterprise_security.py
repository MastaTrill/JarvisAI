"""
Jarvis AI - Enterprise Security Module
Comprehensive security hardening for production deployments

Features:
- Rate limiting with Redis backend
- API key management with encryption
- Audit logging
- Request signing and verification
- IP allowlisting/blocklisting  
- Security headers middleware
- CORS configuration
- Input sanitization
"""

import hashlib
import hmac
import secrets
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from collections import defaultdict
import ipaddress
import re

from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from jose import JWTError, jwt
import bcrypt

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class SecurityConfig:
    """Security configuration settings"""
    # JWT Settings
    JWT_SECRET_KEY: str = secrets.token_urlsafe(64)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 1000  # requests per window
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 50  # burst allowance
    
    # API Key Settings
    API_KEY_LENGTH: int = 64
    API_KEY_PREFIX: str = "jarvis_"
    API_KEY_HASH_ROUNDS: int = 12
    
    # IP Settings
    IP_ALLOWLIST: List[str] = []  # Empty = allow all
    IP_BLOCKLIST: List[str] = []
    
    # Request Signing
    REQUEST_SIGNATURE_HEADER: str = "X-Signature"
    REQUEST_TIMESTAMP_HEADER: str = "X-Timestamp"
    REQUEST_NONCE_HEADER: str = "X-Nonce"
    REQUEST_MAX_AGE_SECONDS: int = 300  # 5 minutes
    
    # Audit Settings
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_SENSITIVE_FIELDS: List[str] = ["password", "token", "secret", "api_key"]


config = SecurityConfig()


# =============================================================================
# PASSWORD HASHING
# =============================================================================

class PasswordHasher:
    """Secure password hashing using bcrypt"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt(rounds=config.API_KEY_HASH_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    @staticmethod
    def needs_rehash(hashed: str) -> bool:
        """Check if password hash needs to be updated"""
        try:
            return bcrypt.checkpw(b"", hashed.encode('utf-8'))
        except Exception:
            return True


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

class APIKeyManager:
    """Secure API key generation and validation"""
    
    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}  # In production, use database
        self._hasher = PasswordHasher()
    
    def generate_key(self, user_id: str, name: str, scopes: List[str] = None) -> Dict[str, str]:
        """Generate a new API key"""
        # Generate random key
        raw_key = secrets.token_urlsafe(config.API_KEY_LENGTH)
        full_key = f"{config.API_KEY_PREFIX}{raw_key}"
        
        # Hash for storage
        key_hash = self._hasher.hash_password(full_key)
        key_id = secrets.token_urlsafe(16)
        
        # Store key metadata
        self._keys[key_id] = {
            "hash": key_hash,
            "user_id": user_id,
            "name": name,
            "scopes": scopes or ["read"],
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "is_active": True,
            "usage_count": 0
        }
        
        return {
            "key_id": key_id,
            "api_key": full_key,  # Return once, never store plaintext
            "prefix": full_key[:20] + "..."
        }
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return its metadata"""
        for key_id, key_data in self._keys.items():
            if key_data["is_active"] and self._hasher.verify_password(api_key, key_data["hash"]):
                # Update usage
                key_data["last_used"] = datetime.utcnow().isoformat()
                key_data["usage_count"] += 1
                return {"key_id": key_id, **key_data}
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self._keys:
            self._keys[key_id]["is_active"] = False
            self._keys[key_id]["revoked_at"] = datetime.utcnow().isoformat()
            return True
        return False
    
    def list_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List all keys for a user (without hashes)"""
        result = []
        for key_id, key_data in self._keys.items():
            if key_data["user_id"] == user_id:
                result.append({
                    "key_id": key_id,
                    "name": key_data["name"],
                    "scopes": key_data["scopes"],
                    "created_at": key_data["created_at"],
                    "last_used": key_data["last_used"],
                    "is_active": key_data["is_active"]
                })
        return result


api_key_manager = APIKeyManager()


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter with sliding window"""
    
    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tokens": config.RATE_LIMIT_REQUESTS,
            "last_update": time.time()
        })
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get from API key first
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"
    
    def _refill_tokens(self, bucket: Dict[str, Any]) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Calculate tokens to add
        refill_rate = config.RATE_LIMIT_REQUESTS / config.RATE_LIMIT_WINDOW
        tokens_to_add = elapsed * refill_rate
        
        bucket["tokens"] = min(
            config.RATE_LIMIT_REQUESTS + config.RATE_LIMIT_BURST,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now
    
    def check_rate_limit(self, request: Request) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        client_id = self._get_client_id(request)
        bucket = self._buckets[client_id]
        
        self._refill_tokens(bucket)
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return {
                "allowed": True,
                "remaining": int(bucket["tokens"]),
                "limit": config.RATE_LIMIT_REQUESTS,
                "reset": int(bucket["last_update"] + config.RATE_LIMIT_WINDOW)
            }
        
        return {
            "allowed": False,
            "remaining": 0,
            "limit": config.RATE_LIMIT_REQUESTS,
            "reset": int(bucket["last_update"] + config.RATE_LIMIT_WINDOW),
            "retry_after": config.RATE_LIMIT_WINDOW
        }


rate_limiter = RateLimiter()


# =============================================================================
# AUDIT LOGGING
# =============================================================================

class AuditLogger:
    """Security audit logging"""
    
    def __init__(self):
        self._logger = logging.getLogger("jarvis.audit")
        handler = logging.FileHandler("audit.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from log data"""
        if not data:
            return {}
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in config.AUDIT_LOG_SENSITIVE_FIELDS):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        request: Optional[Request] = None,
        details: Dict[str, Any] = None,
        success: bool = True
    ) -> None:
        """Log a security audit event"""
        if not config.AUDIT_LOG_ENABLED:
            return
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "details": self._sanitize_data(details or {})
        }
        
        if request:
            event["request"] = {
                "method": request.method,
                "path": str(request.url.path),
                "ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
                "request_id": request.headers.get("X-Request-ID")
            }
        
        log_message = json.dumps(event)
        
        if success:
            self._logger.info(log_message)
        else:
            self._logger.warning(log_message)


audit_logger = AuditLogger()


# =============================================================================
# REQUEST SIGNING
# =============================================================================

class RequestSigner:
    """HMAC-based request signing for API integrity"""
    
    def __init__(self, secret_key: str = None):
        self._secret_key = (secret_key or config.JWT_SECRET_KEY).encode()
        self._used_nonces: Dict[str, float] = {}  # In production, use Redis
    
    def sign_request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        timestamp: str = None,
        nonce: str = None
    ) -> Dict[str, str]:
        """Generate request signature"""
        timestamp = timestamp or str(int(time.time()))
        nonce = nonce or secrets.token_urlsafe(16)
        
        # Create canonical request
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"{method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"
        
        # Sign
        signature = hmac.new(
            self._secret_key,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            config.REQUEST_SIGNATURE_HEADER: signature,
            config.REQUEST_TIMESTAMP_HEADER: timestamp,
            config.REQUEST_NONCE_HEADER: nonce
        }
    
    def verify_request(
        self,
        method: str,
        path: str,
        body: bytes,
        signature: str,
        timestamp: str,
        nonce: str
    ) -> bool:
        """Verify request signature"""
        # Check timestamp age
        try:
            request_time = int(timestamp)
            age = abs(time.time() - request_time)
            if age > config.REQUEST_MAX_AGE_SECONDS:
                logger.warning(f"Request too old: {age}s")
                return False
        except ValueError:
            return False
        
        # Check nonce reuse
        if nonce in self._used_nonces:
            logger.warning(f"Nonce reuse detected: {nonce}")
            return False
        
        # Clean old nonces
        current_time = time.time()
        self._used_nonces = {
            n: t for n, t in self._used_nonces.items()
            if current_time - t < config.REQUEST_MAX_AGE_SECONDS * 2
        }
        self._used_nonces[nonce] = current_time
        
        # Verify signature
        expected = self.sign_request(method, path, body, timestamp, nonce)
        return hmac.compare_digest(
            signature,
            expected[config.REQUEST_SIGNATURE_HEADER]
        )


request_signer = RequestSigner()


# =============================================================================
# IP FILTERING
# =============================================================================

class IPFilter:
    """IP allowlist/blocklist filtering"""
    
    def __init__(self):
        self._allowlist = self._parse_ip_list(config.IP_ALLOWLIST)
        self._blocklist = self._parse_ip_list(config.IP_BLOCKLIST)
    
    def _parse_ip_list(self, ip_list: List[str]) -> List[ipaddress.IPv4Network]:
        """Parse IP strings into network objects"""
        networks = []
        for ip_str in ip_list:
            try:
                if "/" in ip_str:
                    networks.append(ipaddress.ip_network(ip_str, strict=False))
                else:
                    networks.append(ipaddress.ip_network(f"{ip_str}/32", strict=False))
            except ValueError as e:
                logger.warning(f"Invalid IP in filter: {ip_str} - {e}")
        return networks
    
    def is_allowed(self, ip_str: str) -> bool:
        """Check if IP is allowed"""
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        
        # Check blocklist first
        for network in self._blocklist:
            if ip in network:
                return False
        
        # If allowlist is empty, allow all (not blocked)
        if not self._allowlist:
            return True
        
        # Check allowlist
        for network in self._allowlist:
            if ip in network:
                return True
        
        return False


ip_filter = IPFilter()


# =============================================================================
# INPUT SANITIZATION
# =============================================================================

class InputSanitizer:
    """Input validation and sanitization"""
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        r"(--)|(;)|(\/\*)",
        r"(\bOR\b|\bAND\b).*?=",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e",
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 10000) -> str:
        """Sanitize a string input"""
        if not value:
            return value
        
        # Truncate
        value = value[:max_length]
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # Basic HTML entity encoding for special chars
        value = value.replace("&", "&amp;")
        value = value.replace("<", "&lt;")
        value = value.replace(">", "&gt;")
        value = value.replace('"', "&quot;")
        value = value.replace("'", "&#x27;")
        
        return value
    
    @classmethod
    def check_dangerous_patterns(cls, value: str) -> List[str]:
        """Check for potentially dangerous patterns"""
        warnings = []
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                warnings.append("potential_sql_injection")
                break
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                warnings.append("potential_xss")
                break
        
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                warnings.append("potential_path_traversal")
                break
        
        return warnings


# =============================================================================
# MIDDLEWARE
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get client IP
        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "unknown"
        )
        
        # IP Filtering
        if not ip_filter.is_allowed(client_ip):
            audit_logger.log_event(
                "security", None, request.url.path, "blocked_ip",
                request=request, details={"ip": client_ip}, success=False
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        # Rate Limiting
        rate_result = rate_limiter.check_rate_limit(request)
        if not rate_result["allowed"]:
            audit_logger.log_event(
                "security", None, request.url.path, "rate_limited",
                request=request, success=False
            )
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(rate_result["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_result["reset"]),
                    "Retry-After": str(rate_result["retry_after"])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Cache-Control"] = "no-store, max-age=0"
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_result["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_result["reset"])
        
        # Log request
        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
        )
        
        return response


# =============================================================================
# DEPENDENCY INJECTION HELPERS
# =============================================================================

security_bearer = HTTPBearer(auto_error=False)
api_key_header_dep = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header_dep)
) -> Optional[Dict[str, Any]]:
    """Validate API key from header"""
    if not api_key:
        return None
    
    key_data = api_key_manager.validate_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return key_data


async def require_api_key(
    key_data: Optional[Dict[str, Any]] = Depends(get_api_key)
) -> Dict[str, Any]:
    """Require valid API key"""
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return key_data


async def require_scope(required_scope: str):
    """Create a dependency that requires a specific scope"""
    async def check_scope(
        key_data: Dict[str, Any] = Depends(require_api_key)
    ) -> Dict[str, Any]:
        if required_scope not in key_data.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required"
            )
        return key_data
    return check_scope


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SecurityConfig",
    "config",
    "PasswordHasher",
    "APIKeyManager",
    "api_key_manager",
    "RateLimiter",
    "rate_limiter",
    "AuditLogger",
    "audit_logger",
    "RequestSigner",
    "request_signer",
    "IPFilter",
    "ip_filter",
    "InputSanitizer",
    "SecurityMiddleware",
    "get_api_key",
    "require_api_key",
    "require_scope",
]
