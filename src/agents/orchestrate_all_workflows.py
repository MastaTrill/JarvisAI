"""
Unified Orchestration Script for Jarvis AI Platform

This script integrates continuous learning, XAI, multi-modal, and RL workflows.
"""

import subprocess
import sys
import os

# --- Paths to scripts ---
RETRAIN_SCRIPT = "automated_retraining_pipeline.py"
XAI_SCRIPT = "prototype_xai_shap.py"
MULTIMODAL_SCRIPT = "prototype_multimodal.py"
RL_SCRIPT = "prototype_rl_agent.py"

# --- Utility to run a script ---
def run_script(script):
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr)

# --- Main orchestration ---
def main():
    # 1. Continuous Learning Pipeline
    run_script(RETRAIN_SCRIPT)
    # 2. XAI Analysis
    run_script(XAI_SCRIPT)
    # 3. Multi-Modal Learning
    run_script(MULTIMODAL_SCRIPT)
    # 4. Reinforcement Learning Agent
    run_script(RL_SCRIPT)
    print("\nAll workflows completed.")

if __name__ == "__main__":
    main()
