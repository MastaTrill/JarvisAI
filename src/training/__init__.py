"""
Training package for Jarvis AI Project.

This package contains training utilities and trainer classes.
"""

try:
    from .trainer import Trainer
except ImportError:
    try:
        # Fallback to simple trainer if torch is not available
        from .simple_trainer import SimpleTrainer as Trainer
    except ImportError:
        # Final fallback to numpy trainer
        from .numpy_trainer import NumpyTrainer as Trainer

__all__ = ['Trainer']
