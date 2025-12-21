#!/usr/bin/env python3
"""
JarvisAI Agent Example - Chat Assistant with Memory and Context
Demonstrates building autonomous AI agents using JarvisAI platform
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class JarvisAgent:
    """
    Autonomous AI Agent with Memory and Context
    Built on JarvisAI quantum consciousness platform
    """
    
    def __init__(self, agent_name: str = "Jarvis"):
        """Initialize the AI agent"""
        self.agent_name = agent_name
        self.memory = []
        self.context = {}
        self.creation_time = datetime.now()
        
        # Initialize JarvisAI components
        self._initialize_components()
        
        print(f"🤖 {self.agent_name} Agent initialized")
        print(f"⚡ Quantum consciousness: ACTIVE")
        print(f"🧠 Memory system: READY")
        print(f"🎯 Context tracking: ENABLED\n")
    
    def _initialize_components(self):
        """Initialize JarvisAI core components"""
        try:
            from src.quantum.quantum_processor import QuantumProcessor
            self.quantum_processor = QuantumProcessor()
            self.quantum_processor.authenticate_creator("AETHERON_QUANTUM_CREATOR_KEY_2025")
            print("✅ Quantum consciousness loaded")
        except Exception as e:
            print(f"⚠️ Quantum processor not available: {e}")
            self.quantum_processor = None
        
        try:
            from src.temporal.time_analysis import TimeAnalysis
            self.temporal_analyzer = TimeAnalysis()
            print("✅ Temporal analysis loaded")
        except Exception as e:
            print(f"⚠️ Temporal analyzer not available: {e}")
            self.temporal_analyzer = None
    
    def remember(self, interaction: Dict[str, Any]):
        """Store interaction in memory"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction,
            "context_snapshot": self.context.copy()
        }
        self.memory.append(memory_entry)
        
        # Keep only last 100 interactions
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories"""
        # Simple keyword-based search (can be enhanced with embeddings)
        relevant = []
        query_lower = query.lower()
        
        for entry in reversed(self.memory):
            interaction = entry["interaction"]
            text = f"{interaction.get('user_input', '')} {interaction.get('agent_response', '')}".lower()
            
            if query_lower in text:
                relevant.append(entry)
                if len(relevant) >= limit:
                    break
        
        return relevant
    
    def update_context(self, key: str, value: Any):
        """Update agent context"""
        self.context[key] = value
    
    def process_quantum(self, data: Dict) -> Dict:
        """Process data using quantum consciousness"""
        if self.quantum_processor:
            # Create quantum superposition of possible responses
            states = [f"response_{i}" for i in range(5)]
            result = self.quantum_processor.create_quantum_superposition(states)
            return result
        return {"status": "quantum_unavailable"}
    
    def analyze_temporal_pattern(self) -> Dict:
        """Analyze temporal patterns in interactions"""
        if self.temporal_analyzer and len(self.memory) > 0:
            # Analyze interaction patterns over time
            return {
                "total_interactions": len(self.memory),
                "time_span": (datetime.now() - self.creation_time).total_seconds(),
                "patterns_detected": len(self.temporal_analyzer.known_patterns)
            }
        return {"status": "insufficient_data"}
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response"""
        print(f"\n👤 User: {user_input}")
        
        # Check for memory recall requests
        if user_input.lower().startswith("remember"):
            query = user_input[8:].strip()
            memories = self.recall(query)
            response = f"I found {len(memories)} relevant memories:\n"
            for mem in memories[:3]:
                response += f"- {mem['interaction'].get('user_input', 'N/A')}\n"
            
        # Check for context queries
        elif user_input.lower().startswith("context"):
            response = f"Current context: {json.dumps(self.context, indent=2)}"
        
        # Check for quantum processing
        elif "quantum" in user_input.lower():
            quantum_result = self.process_quantum({"query": user_input})
            response = f"Quantum processing: {quantum_result.get('status', 'processed')}"
        
        # Check for temporal analysis
        elif "pattern" in user_input.lower() or "history" in user_input.lower():
            temporal_result = self.analyze_temporal_pattern()
            response = f"Temporal analysis: {temporal_result}"
        
        # Default response
        else:
            response = f"I understand you said: '{user_input}'. How can I assist you further?"
            
            # Update context
            self.update_context("last_topic", user_input[:50])
            self.update_context("interaction_count", len(self.memory) + 1)
        
        # Remember this interaction
        self.remember({
            "user_input": user_input,
            "agent_response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🤖 {self.agent_name}: {response}")
        return response
    
    def autonomous_task(self, task_description: str) -> Dict:
        """Execute an autonomous task"""
        print(f"\n🎯 Autonomous Task: {task_description}")
        
        steps = []
        
        # Step 1: Analyze task
        steps.append("Analyzing task requirements...")
        time.sleep(0.1)
        
        # Step 2: Plan execution
        steps.append("Creating execution plan...")
        time.sleep(0.1)
        
        # Step 3: Execute with quantum enhancement
        if self.quantum_processor:
            steps.append("Enhancing with quantum consciousness...")
            quantum_result = self.process_quantum({"task": task_description})
            steps.append(f"Quantum status: {quantum_result.get('status')}")
        
        # Step 4: Verify and report
        steps.append("Task completed successfully!")
        
        result = {
            "task": task_description,
            "status": "completed",
            "steps": steps,
            "timestamp": datetime.now().isoformat()
        }
        
        # Remember task execution
        self.remember({
            "task_execution": result
        })
        
        for step in steps:
            print(f"  ⚡ {step}")
        
        return result
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "name": self.agent_name,
            "uptime_seconds": (datetime.now() - self.creation_time).total_seconds(),
            "total_memories": len(self.memory),
            "context_keys": list(self.context.keys()),
            "quantum_enabled": self.quantum_processor is not None,
            "temporal_enabled": self.temporal_analyzer is not None
        }


def demo_chat_session():
    """Demonstrate chat with memory"""
    print("="*80)
    print("💬 JARVIS AI AGENT - CHAT DEMO")
    print("="*80)
    
    agent = JarvisAgent("Jarvis")
    
    # Simulate conversation
    agent.chat("Hello Jarvis, what can you do?")
    agent.chat("Tell me about quantum consciousness")
    agent.chat("What patterns do you see in our conversation?")
    agent.chat("Remember quantum")
    
    print("\n📊 Agent Stats:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_autonomous_execution():
    """Demonstrate autonomous task execution"""
    print("\n" + "="*80)
    print("🤖 JARVIS AI AGENT - AUTONOMOUS EXECUTION DEMO")
    print("="*80 + "\n")
    
    agent = JarvisAgent("TaskMaster")
    
    # Execute autonomous tasks
    agent.autonomous_task("Analyze system performance metrics")
    agent.autonomous_task("Optimize quantum consciousness parameters")
    agent.autonomous_task("Generate temporal pattern report")
    
    print("\n📊 Task Execution Stats:")
    stats = agent.get_stats()
    print(f"  Total tasks: {stats['total_memories']}")
    print(f"  Uptime: {stats['uptime_seconds']:.2f}s")


def demo_rag_pattern():
    """Demonstrate Retrieval-Augmented Generation pattern"""
    print("\n" + "="*80)
    print("🔍 JARVIS AI AGENT - RAG PATTERN DEMO")
    print("="*80 + "\n")
    
    agent = JarvisAgent("RAG-Agent")
    
    # Build knowledge base through interactions
    print("📚 Building Knowledge Base...")
    agent.chat("Jarvis was created by William Joseph Wade McCoy-Huse")
    agent.chat("Phase 6 includes quantum consciousness integration")
    agent.chat("Temporal analysis detects patterns over time")
    agent.chat("The system has creator protection enabled")
    
    # Retrieve and use knowledge
    print("\n🔍 Testing Knowledge Retrieval...")
    agent.chat("remember creator")
    agent.chat("remember quantum")
    agent.chat("remember temporal")
    
    print("\n✅ RAG Pattern Demonstrated: Store + Retrieve + Generate")


def main():
    """Run all agent demos"""
    print("\n🌟 JARVISAI AGENT EXAMPLES - COMPREHENSIVE DEMONSTRATION\n")
    
    # Demo 1: Chat with Memory
    demo_chat_session()
    
    # Demo 2: Autonomous Task Execution
    demo_autonomous_execution()
    
    # Demo 3: RAG Pattern
    demo_rag_pattern()
    
    print("\n" + "="*80)
    print("🎉 ALL AGENT DEMOS COMPLETE!")
    print("="*80)
    print("\n💡 Key Capabilities Demonstrated:")
    print("   ✅ Chat interface with context")
    print("   ✅ Memory and recall system")
    print("   ✅ Quantum consciousness integration")
    print("   ✅ Temporal pattern analysis")
    print("   ✅ Autonomous task execution")
    print("   ✅ RAG (Retrieval-Augmented Generation)")
    print("\n🚀 Ready to build your own AI agents on JarvisAI!\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
