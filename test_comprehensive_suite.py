#!/usr/bin/env python3
"""
Comprehensive Test Suite for Aetheron AI Platform - Advanced Features.
Tests all major components and features of the enhanced Jarvis system.
"""

import sys
from pathlib import Path
import logging
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test_suite():
    """Run comprehensive test suite for all advanced features."""
    logger.info("🧪 AETHERON AI PLATFORM - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Test 1: Basic Training System
    logger.info("\n1️⃣ Testing Basic Training System...")
    try:
        result = subprocess.run([sys.executable, "simple_train.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            test_results['basic_training'] = "✅ PASSED"
            logger.info("   ✅ Basic training system works correctly")
        else:
            test_results['basic_training'] = "❌ FAILED"
            logger.error(f"   ❌ Basic training failed: {result.stderr}")
    except Exception as e:
        test_results['basic_training'] = f"❌ ERROR: {e}"
        logger.error(f"   ❌ Basic training error: {e}")
    
    # Test 2: Advanced Features Demo
    logger.info("\n2️⃣ Testing Advanced Features Demo...")
    try:
        result = subprocess.run([sys.executable, "demo_advanced_features.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            test_results['advanced_features'] = "✅ PASSED"
            logger.info("   ✅ Advanced features demo completed successfully")
        else:
            test_results['advanced_features'] = "❌ FAILED"
            logger.error(f"   ❌ Advanced features failed: {result.stderr}")
    except Exception as e:
        test_results['advanced_features'] = f"❌ ERROR: {e}"
        logger.error(f"   ❌ Advanced features error: {e}")
    
    # Test 3: Simplified Advanced Demo
    logger.info("\n3️⃣ Testing Simplified Advanced Demo...")
    try:
        result = subprocess.run([sys.executable, "demo_simplified_advanced.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            test_results['simplified_demo'] = "✅ PASSED"
            logger.info("   ✅ Simplified advanced demo completed successfully")
        else:
            test_results['simplified_demo'] = "❌ FAILED"
            logger.error(f"   ❌ Simplified demo failed: {result.stderr}")
    except Exception as e:
        test_results['simplified_demo'] = f"❌ ERROR: {e}"
        logger.error(f"   ❌ Simplified demo error: {e}")
    
    # Test 4: Integration Demo
    logger.info("\n4️⃣ Testing Integration Demo...")
    try:
        result = subprocess.run([sys.executable, "demo_integration_advanced.py"], 
                              capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            test_results['integration_demo'] = "✅ PASSED"
            logger.info("   ✅ Integration demo completed successfully")
        else:
            test_results['integration_demo'] = "❌ FAILED"
            logger.error(f"   ❌ Integration demo failed: {result.stderr}")
    except Exception as e:
        test_results['integration_demo'] = f"❌ ERROR: {e}"
        logger.error(f"   ❌ Integration demo error: {e}")
    
    # Test 5: API Server (Quick Test)
    logger.info("\n5️⃣ Testing Enhanced API Server...")
    api_process = None
    try:
        # Start API server in background
        api_process = subprocess.Popen([sys.executable, "api_enhanced.py"])
        time.sleep(3)  # Give server time to start
        
        # Test basic health check (simplified - just check if server starts)
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                test_results['api_server'] = "✅ PASSED"
                logger.info("   ✅ Enhanced API server is working")
            else:
                test_results['api_server'] = "❌ FAILED"
                logger.error("   ❌ API server health check failed")
        except ImportError:
            # If requests is not available, just check if process started
            if api_process.poll() is None:
                test_results['api_server'] = "✅ PASSED (startup only)"
                logger.info("   ✅ Enhanced API server started (requests module not available)")
            else:
                test_results['api_server'] = "❌ FAILED"
                logger.error("   ❌ API server failed to start")
        
        # Cleanup
        if api_process:
            api_process.terminate()
            api_process.wait(timeout=5)
        
    except Exception as e:
        test_results['api_server'] = f"❌ ERROR: {e}"
        logger.error(f"   ❌ API server error: {e}")
        if api_process:
            try:
                api_process.terminate()
            except:
                pass
    
    # Test 6: Core Module Imports
    logger.info("\n6️⃣ Testing Core Module Imports...")
    import_tests = [
        ("Neural Networks", "src.models.advanced_neural_network", "AdvancedNeuralNetwork"),
        ("Data Pipeline", "src.data.advanced_data_pipeline", "AdvancedDataPipeline"),
        ("Training System", "src.training.advanced_training_system", "AdvancedTrainingSystem"),
        ("Data Augmentation", "src.augmentation.data_augmenter", "DataAugmentationPipeline"),
        ("Model Validation", "src.validation.simple_validator", "SimpleValidator"),
        ("MLFlow Integration", "src.tracking.mlflow_integration", "MLFlowTracker")
    ]
    
    import_results = []
    for name, module_path, class_name in import_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            import_results.append(f"   ✅ {name}")
        except Exception as e:
            import_results.append(f"   ❌ {name}: {e}")
    
    test_results['module_imports'] = "✅ PASSED" if all("✅" in r for r in import_results) else "❌ FAILED"
    logger.info("   Module import results:")
    for result in import_results:
        logger.info(result)
    
    # Test Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUITE SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for result in test_results.values() if "✅" in result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        logger.info(f"   {test_name.replace('_', ' ').title()}: {result}")
    
    logger.info(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Aetheron AI Platform is fully functional.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Review the errors above.")
        return False


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
