"""AdvancedFeatures wrapper for multimodal AI."""

from src.ai.multimodal_ai import MultimodalAI as _MultimodalAI

class MultimodalAI(_MultimodalAI):
    """Compatibility wrapper to expose the multimodal AI helper in advanced_features."""
    pass
