"""
Demo: Multi-agent collaboration and orchestration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from agents.advanced_agent import AdvancedAgent
from agents.orchestrator import AgentOrchestrator

# Dummy LLM and task queue for demonstration
class DummyLLM:
    def generate(self, prompt):
        return f"[LLM] Response to: {prompt}"

class DummyTaskQueue:
    def enqueue(self, task_desc):
        return f"task-{hash(task_desc) % 10000}"

def main():
    llm = DummyLLM()
    task_queue = DummyTaskQueue()
    agent1 = AdvancedAgent("Alpha", llm=llm, task_queue=task_queue)
    agent2 = AdvancedAgent("Beta")
    orchestrator = AgentOrchestrator([agent1, agent2])

    print("\n--- Multi-Agent Broadcast ---")
    responses = orchestrator.broadcast("Analyze system status")
    for name, resp in responses.items():
        print(f"{name}: {resp}")

    print("\n--- Delegate Task to Alpha ---")
    print(orchestrator.delegate("Alpha", "Optimize quantum parameters"))
    print("\n--- Delegate Task to Beta ---")
    print(orchestrator.delegate("Beta", "Summarize memory"))

if __name__ == "__main__":
    # Run as a module to support relative imports
    import runpy
    import os
    if os.getcwd().endswith('agents'):
        main()
    else:
        import sys
        import subprocess
        subprocess.run([sys.executable, '-m', 'agents.demo_multi_agent'])
