"""
Models package for Jarvis AI Project.

This package contains neural network models and related utilities.
"""

try:
    from .neural_network import SimpleNeuralNetwork
except ImportError:
    try:
        # Fallback to scikit-learn based implementation
        from .simple_neural_network import SimpleNeuralNetwork
    except ImportError:
        # Final fallback to numpy implementation
        from .numpy_neural_network import SimpleNeuralNetwork

# Import advanced models
try:
    from .advanced_neural_network import AdvancedNeuralNetwork
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = ['SimpleNeuralNetwork']
if ADVANCED_AVAILABLE:
    __all__.append('AdvancedNeuralNetwork')

