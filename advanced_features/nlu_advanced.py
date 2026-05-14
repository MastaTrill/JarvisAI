"""
Advanced Natural Language Understanding (contextual memory, reasoning).
"""


class AdvancedNLU:
    """Basic advanced NLU with contextual memory and reasoning."""
    def understand(self, text, context):
        """
        Simulate NLU with context and simple reasoning.
        Args:
            text: input string
            context: dict with 'memory' key (list of strings)
        Returns:
            dict: response with context summary
        """
        memory = context.get('memory', [])
        response = {
            'input': text,
            'context_used': memory[-1] if memory else None,
            'reasoning': f"Processed '{text}' with context."
        }
        return response
