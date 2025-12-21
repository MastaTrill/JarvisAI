"""
Multi-agent orchestration for JarvisAI.
Allows agents to communicate, collaborate, and delegate tasks.
"""
from typing import List, Dict, Any

class AgentOrchestrator:
    def __init__(self, agents: List):
        self.agents = agents

    def broadcast(self, message: str) -> Dict[str, str]:
        """Send a message to all agents and collect responses."""
        responses = {}
        for agent in self.agents:
            try:
                responses[agent.name] = agent.act(message)
            except Exception as e:
                responses[agent.name] = f"Error: {e}"
        return responses

    def delegate(self, agent_name: str, task: str) -> str:
        """Send a task to a specific agent."""
        for agent in self.agents:
            if agent.name == agent_name:
                return agent.act(task)
        return f"Agent '{agent_name}' not found."
