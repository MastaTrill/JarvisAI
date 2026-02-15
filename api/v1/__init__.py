"""
Jarvis AI - API v1 Package
Standardized REST API following OpenAPI 3.0 specification
"""

from .routes import router, APIResponse, PaginatedResponse, ErrorResponse
from .routes import create_response, create_paginated_response, create_error_response

__all__ = [
    "router",
    "APIResponse",
    "PaginatedResponse",
    "ErrorResponse",
    "create_response",
    "create_paginated_response",
    "create_error_response",
]
