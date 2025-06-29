"""
Data package for Jarvis AI Project.

This package contains data processing utilities and loaders.
"""

try:
    from .processor import DataProcessor
except ImportError:
    # Fallback to numpy-only processor
    from .numpy_processor import DataProcessor

__all__ = ['DataProcessor']