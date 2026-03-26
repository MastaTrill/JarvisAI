"""
Jarvis AI - API Package
Versioned API structure with standardized patterns
"""

from fastapi import APIRouter

# Main API router that aggregates all versions
api_router = APIRouter()

# Import and include versioned routers
from .v1 import router as v1_router

api_router.include_router(v1_router)

# Patch: Export get_current_user from main_api for compatibility
try:
    from main_api import get_current_user
except ImportError:

    def get_current_user():
        raise NotImplementedError("get_current_user not available")


__all__ = ["api_router", "get_current_user"]
