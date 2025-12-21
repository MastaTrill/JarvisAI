#!/usr/bin/env python3
"""
Reality Check Test Suite - What Actually Works?
Tests each module systematically to determine actual vs. theoretical capabilities.
"""

import sys
import importlib

# Test results tracking
results = {
    'working': [],
    'broken': [],
    'missing': []
}

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        results['working'].append((module_name, description))
        return True, "✅"
    except ImportError as e:
        if "No module named" in str(e):
            results['missing'].append((module_name, description, str(e)))
            return False, "❌ MISSING"
        else:
            results['broken'].append((module_name, description, str(e)))
            return False, "⚠️ BROKEN"
    except (AttributeError, ValueError, TypeError, RuntimeError) as e:
        results['broken'].append((module_name, description, str(e)))
        return False, "⚠️ ERROR"

def test_src_module(module_path, description):
    """Test if a source module can be imported"""
    module_name = f"src.{module_path}"
    return test_import(module_name, description)

print("=" * 80)
print("🔍 JARVIS REALITY CHECK - Testing Actual Capabilities")
print("=" * 80)

# Test Core Dependencies
print("\n📦 Core Dependencies:")
print("-" * 80)
test_import("numpy", "NumPy - Core numerical computing")
test_import("pandas", "Pandas - Data manipulation")
test_import("matplotlib", "Matplotlib - Visualization")
test_import("seaborn", "Seaborn - Statistical visualization")
test_import("sklearn", "Scikit-learn - ML library")
test_import("yaml", "PyYAML - Config files")

# Test Advanced Dependencies
print("\n🚀 Advanced ML Dependencies:")
print("-" * 80)
test_import("torch", "PyTorch - Deep learning")
test_import("tensorflow", "TensorFlow - Deep learning")
test_import("transformers", "Transformers - NLP models")
test_import("mlflow", "MLflow - Experiment tracking")

# Test Core Jarvis Modules
print("\n🧠 Core Jarvis Modules:")
print("-" * 80)
test_src_module("models.numpy_neural_network", "Numpy Neural Network")
test_src_module("training.simple_trainer", "Simple Trainer")
test_src_module("data.numpy_processor", "Numpy Data Processor")
test_src_module("validation.simple_validator", "Simple Validator")

# Test Next-Gen Modules
print("\n🌟 Next-Generation AI Modules:")
print("-" * 80)
test_src_module("neuromorphic.neuromorphic_brain", "Neuromorphic AI Brain")
test_src_module("quantum.quantum_neural_networks", "Quantum Neural Networks")
test_src_module("cv.advanced_computer_vision", "Advanced Computer Vision")
test_src_module("biotech.biotech_ai", "Biotech AI Module")
test_src_module("prediction.prediction_oracle", "Prediction Oracle")
test_src_module("robotics.autonomous_robotics", "Autonomous Robotics")
test_src_module("distributed.hyperscale_distributed_ai", "Hyperscale Distributed AI")
test_src_module("space.space_ai_mission_control", "Space AI Mission Control")

# Test Other Advanced Modules
print("\n🔬 Other Advanced Modules:")
print("-" * 80)
test_src_module("consciousness.consciousness_engine", "Consciousness Engine")
test_src_module("temporal.temporal_engine", "Temporal Engine")
test_src_module("cosmic.cosmic_intelligence", "Cosmic Intelligence")
test_src_module("transcendent.transcendent_ai", "Transcendent AI")

# Summary Report
print("\n" + "=" * 80)
print("📊 SUMMARY REPORT")
print("=" * 80)

print(f"\n✅ Working Modules ({len(results['working'])}):")
for module, desc in results['working']:
    print(f"  • {desc}")

print(f"\n❌ Missing Dependencies ({len(results['missing'])}):")
for module, desc, error in results['missing']:
    print(f"  • {desc}")
    print(f"    ({module})")

print(f"\n⚠️ Broken/Error Modules ({len(results['broken'])}):")
for module, desc, error in results['broken']:
    print(f"  • {desc}")
    print(f"    Error: {error[:100]}")

# Calculate percentage
total = len(results['working']) + len(results['missing']) + len(results['broken'])
working_pct = (len(results['working']) / total * 100) if total > 0 else 0

print("\n" + "=" * 80)
print(f"🎯 REALITY CHECK: {working_pct:.1f}% of modules are currently working")
print("=" * 80)

# Exit with status
sys.exit(0 if len(results['broken']) == 0 else 1)
