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

__all__ = ["api_router"]
