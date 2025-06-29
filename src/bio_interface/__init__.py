"""
🧬 Biological Integration Interface Module

This module provides advanced brain-computer interface capabilities
for seamless human-AI collaboration with full safety protocols.

Created for the Jarvis/Aetheron AI Platform - Next Evolutionary Phase.
"""

try:
    from .neural_bridge import NeuralBridge
    from .thought_translation import ThoughtTranslator
    from .memory_enhancement import MemoryEnhancer
    from .safety_protocols import NeuralSafetySystem
    
    __all__ = [
        'NeuralBridge',
        'ThoughtTranslator', 
        'MemoryEnhancer',
        'NeuralSafetySystem'
    ]
except ImportError as e:
    print(f"Warning: Some bio_interface modules could not be imported: {e}")
    __all__ = []
