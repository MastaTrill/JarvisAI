"""
Inference package for Jarvis AI Project.

This package contains prediction utilities and inference classes.
"""

try:
    from .predictor import ModelPredictor
except ImportError:
    # If torch-based predictor fails, we'll skip it
    ModelPredictor = None

__all__ = ['ModelPredictor'] if ModelPredictor else []
