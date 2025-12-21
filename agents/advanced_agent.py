"""
Example advanced agent using LLM and task queue hooks.
"""
from agents.base import AgentBase

class AdvancedAgent(AgentBase):
    def __init__(self, name: str, llm=None, task_queue=None):
        super().__init__(name)
        self.llm = llm
        self.task_queue = task_queue

    def act(self, user_input: str) -> str:
        # Example: Use LLM if available
        if self.llm:
            response = self.llm.generate(user_input)
        else:
            response = f"{self.name} received: {user_input}"
        self.remember({"user_input": user_input, "agent_response": response})
        return response

    def run_task(self, task_desc: str) -> str:
        if self.task_queue:
            task_id = self.task_queue.enqueue(task_desc)
            return f"Task '{task_desc}' enqueued with ID {task_id}"
        return "No task queue available."
