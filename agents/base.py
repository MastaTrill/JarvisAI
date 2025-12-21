"""
Base class for all JarvisAI agents.
Provides memory, context, and extensibility hooks for LLM, task queue, and collaboration.
"""
from datetime import datetime
from typing import List, Dict, Any

class AgentBase:
    def __init__(self, name: str):
        self.name = name
        self.memory = []
        self.context = {}
        self.creation_time = datetime.now()

    def remember(self, interaction: Dict[str, Any]):
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction,
            "context_snapshot": self.context.copy()
        })
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        query_lower = query.lower()
        relevant = []
        for entry in reversed(self.memory):
            text = str(entry["interaction"]).lower()
            if query_lower in text:
                relevant.append(entry)
                if len(relevant) >= limit:
                    break
        return relevant

    def update_context(self, key: str, value: Any):
        self.context[key] = value

    def act(self, user_input: str) -> str:
        """Override in subclass: process input and return response."""
        raise NotImplementedError
