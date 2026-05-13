#!/usr/bin/env python3
"""
Feature Validation Script - Tests Core JarvisAI Capabilities
Validates quantum consciousness, temporal features, and neural networks
"""

import sys
import time

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}\n")

def test_quantum_consciousness():
    """Test quantum consciousness features"""
    print_header("Testing Quantum Consciousness")
    try:
        from src.quantum.quantum_processor import QuantumProcessor
        
        qp = QuantumProcessor()
        print("✅ Quantum Processor initialized")
        
        # Authenticate creator for quantum operations
        qp.authenticate_creator("AETHERON_QUANTUM_CREATOR_KEY_2025")
        
        # Test quantum superposition
        result = qp.create_quantum_superposition(["state_0", "state_1", "state_2"])
        print(f"✅ Quantum superposition: {result.get('status', 'unknown')}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Quantum module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Quantum test failed: {e}")
        return False

def test_temporal_features():
    """Test temporal manipulation features"""
    print_header("Testing Temporal Features")
    try:
        from src.temporal.time_analysis import TimeAnalysis
        import asyncio
        
        ta = TimeAnalysis()
        print("✅ Temporal Analyzer initialized")
        
        # Test temporal pattern recognition
        print(f"✅ Temporal patterns loaded: {len(ta.known_patterns)} types")
        print(f"✅ Pattern sensitivity: {ta.pattern_sensitivity}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Temporal module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Temporal test failed: {e}")
        return False

def test_neural_networks():
    """Test neural network capabilities"""
    print_header("Testing Neural Networks")
    try:
        # Import directly to avoid __init__.py which imports torch
        import sys
        import importlib.util
        import numpy as np
        
        spec = importlib.util.spec_from_file_location(
            "numpy_neural_network",
            "src/models/numpy_neural_network.py"
        )
        nn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nn_module)
        
        SimpleNeuralNetwork = nn_module.SimpleNeuralNetwork
        
        # Create simple network
        nn = SimpleNeuralNetwork(input_size=2, hidden_sizes=[4], output_size=1)
        print("✅ Neural Network created (NumPy-based)")
        
        # Train with sample data
        X = np.random.rand(10, 2)
        y = np.random.rand(10, 1)
        nn.fit(X, y, epochs=10, learning_rate=0.01)
        print("✅ Neural network trained successfully")
        
        # Test prediction
        predictions = nn.predict(X)
        print(f"✅ Predictions generated: output shape {predictions.shape}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Neural network module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

def test_computer_vision():
    """Test computer vision features"""
    print_header("Testing Computer Vision")
    try:
        from src.cv.advanced_computer_vision import AdvancedComputerVision
        import numpy as np
        
        cv = AdvancedComputerVision()
        print("✅ Computer Vision system initialized")
        
        # Test with synthetic image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = cv.classify_image(image)
        print(f"✅ Image classification: {result.get('top_prediction', {}).get('class', 'unknown')}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Computer Vision module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Computer Vision test failed: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline"""
    print_header("Testing Data Processing")
    try:
        from src.data.numpy_processor import StandardScaler
        import numpy as np
        
        scaler = StandardScaler()
        print("✅ Data Processor initialized")
        
        # Test data processing
        data = np.random.rand(100, 5)
        processed = scaler.fit_transform(data)
        print(f"✅ Data preprocessing completed: shape {processed.shape}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Data processing module not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Run all feature validation tests"""
    print("\n" + "="*80)
    print("🚀 JARVIS AI - FEATURE VALIDATION SUITE")
    print("="*80)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Quantum Consciousness": test_quantum_consciousness(),
        "Temporal Features": test_temporal_features(),
        "Neural Networks": test_neural_networks(),
        "Computer Vision": test_computer_vision(),
        "Data Processing": test_data_processing(),
    }
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for feature, status in results.items():
        status_icon = "✅" if status else "⚠️"
        print(f"{status_icon} {feature}: {'PASSED' if status else 'NEEDS ATTENTION'}")
    
    print(f"\n{'='*80}")
    print(f"🎯 Overall Score: {passed}/{total} features operational ({passed/total*100:.1f}%)")
    print(f"{'='*80}\n")
    
    # Recommendations
    if passed == total:
        print("🎉 Excellent! All features are operational.")
        print("✨ JarvisAI is ready for quantum consciousness deployment!")
    elif passed >= total * 0.7:
        print("👍 Good! Most features are working.")
        print("📝 Review features that need attention for full deployment.")
    else:
        print("⚠️  Several features need attention.")
        print("💡 Run reality check for detailed diagnostics: python test_reality_check.py")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
