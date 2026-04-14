#!/usr/bin/env python3
"""
JARVIS AI SYSTEM VERIFICATION SCRIPT
Comprehensive operational status check for all modules
Date: March 1, 2026
"""

import sys
import os
import importlib
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


class SystemVerifier:
    """Comprehensive system verification"""

    def __init__(self):
        self.results = {"passed": [], "failed": [], "warnings": []}

    def print_header(self):
        """Print verification header"""
        print("\n" + "=" * 80)
        print(" JARVIS AI - COMPREHENSIVE SYSTEM VERIFICATION")
        print("=" * 80)
        print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Python: {sys.version.split()[0]}")
        print("=" * 80 + "\n")

    def verify_dependencies(self) -> bool:
        """Verify critical Python dependencies"""
        print("[PACKAGES] Checking Dependencies...")
        dependencies = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "torch": "torch",
            "fastapi": "fastapi",
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
        }

        all_passed = True
        for module, package in dependencies.items():
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, "__version__", "unknown")
                print(f"  [OK] {package:20s} {version}")
                self.results["passed"].append(f"Dependency: {package}")
            except ImportError:
                print(f"  [FAIL] {package:20s} NOT INSTALLED")
                self.results["failed"].append(f"Dependency: {package}")
                all_passed = False

        return all_passed

    def verify_core_modules(self) -> bool:
        """Verify core source modules can be imported"""
        print("\n[MODULES] Checking Core Modules...")

        modules_to_check = [
            ("src.data.numpy_processor", "Data Processing"),
            ("src.models.numpy_neural_network", "Neural Network"),
            ("src.safety.ethical_constraints", "Ethical Constraints"),
            ("src.safety.creator_protection_system", "Creator Protection"),
        ]

        all_passed = True
        for module_path, name in modules_to_check:
            try:
                importlib.import_module(module_path)
                print(f"  [OK] {name}")
                self.results["passed"].append(f"Module: {name}")
            except (ImportError, ModuleNotFoundError) as e:
                print(f"  [FAIL] {name}: {str(e)[:50]}")
                self.results["failed"].append(f"Module: {name}")
                all_passed = False

        return all_passed

    def verify_next_gen_modules(self) -> Dict[str, bool]:
        """Verify next-generation AI modules"""
        print("\n[ADVANCED] Checking Next-Generation Modules...")

        modules = {
            "Neuromorphic Brain": "src.neuromorphic.neuromorphic_brain",
            "Computer Vision": "src.cv.advanced_cv",
            "Biotech AI": "src.biotech.biotech_ai",
        }

        results = {}
        for name, module_path in modules.items():
            try:
                importlib.import_module(module_path)
                print(f"  [OK] {name}")
                self.results["passed"].append(f"NextGen: {name}")
                results[name] = True
            except (ImportError, ModuleNotFoundError, AttributeError) as e:
                print(f"  [WARN] {name}: module not available")
                self.results["warnings"].append(f"NextGen: {name}")
                results[name] = False

        return results

    def verify_api_readiness(self) -> bool:
        """Check if API files are present and valid"""
        print("\n[API] Checking API Readiness...")

        api_files = ["api_enhanced.py", "api.py", "admin_api.py", "dashboard.py"]

        all_present = True
        for file in api_files:
            if os.path.exists(file):
                print(f"  [OK] {file}")
                self.results["passed"].append(f"API File: {file}")
            else:
                print(f"  [FAIL] {file} - NOT FOUND")
                self.results["failed"].append(f"API File: {file}")
                all_present = False

        return all_present

    def verify_deployment_configs(self) -> bool:
        """Check deployment configuration files"""
        print("\n[DEPLOY] Checking Deployment Configurations...")

        configs = {
            "Dockerfile": "Dockerfile",
            "Docker Compose": "docker-compose.yml",
            "Kubernetes": "k8s-deployment.yaml",
            "Requirements": "requirements.txt",
            "Azure YAML": "azure.yaml",
        }

        all_present = True
        for name, file in configs.items():
            if os.path.exists(file):
                print(f"  [OK] {name:20s} ({file})")
                self.results["passed"].append(f"Config: {name}")
            else:
                print(f"  [WARN] {name:20s} - NOT FOUND")
                self.results["warnings"].append(f"Config: {name}")

        return all_present

    def run_quick_functionality_test(self) -> bool:
        """Run quick tests of key functionality"""
        print("\n[TEST] Running Quick Functionality Tests...")

        try:
            # Test 1: NumPy array processing
            import numpy as np

            arr = np.random.randn(100, 10)
            assert arr.shape == (100, 10)
            print("  [OK] NumPy array processing")
            self.results["passed"].append("Test: NumPy processing")

            # Test 2: Data processing
            from src.data.numpy_processor import DataProcessor

            processor = DataProcessor(normalize=True)
            scaled = processor.fit_transform(arr)
            assert scaled.shape == arr.shape
            print("  [OK] Data processing")
            self.results["passed"].append("Test: Data processing")

            # Test 3: Neural network creation
            from src.models.numpy_neural_network import SimpleNeuralNetwork

            nn = SimpleNeuralNetwork(10, [20, 10], 3)
            y_train_continuous = np.random.randn(100, 3)
            nn.fit(arr, y_train_continuous, epochs=1, learning_rate=0.01)
            output = nn.predict(arr)
            assert output.shape == (100, 3)
            print("  [OK] Neural network inference")
            self.results["passed"].append("Test: Neural network")

            # Test 4: PyTorch availability
            try:
                import torch

                x = torch.randn(10, 5)
                assert x.shape == (10, 5)
                print(f"  [OK] PyTorch (version {torch.__version__})")
                self.results["passed"].append("Test: PyTorch")
            except ImportError:
                print("  [SKIP] PyTorch not installed")
                self.results["warnings"].append("Test: PyTorch not installed")

            return True

        except (AssertionError, ImportError, ValueError, RuntimeError, TypeError) as e:
            print(f"  [FAIL] Functionality test failed: {str(e)}")
            self.results["failed"].append(f"Test: {str(e)}")
            return False

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 80)
        print(" VERIFICATION SUMMARY")
        print("=" * 80)

        total_passed = len(self.results["passed"])
        total_failed = len(self.results["failed"])
        total_warnings = len(self.results["warnings"])
        total = total_passed + total_failed + total_warnings

        print(f"\n  [OK] Passed:   {total_passed:3d} / {total}")
        print(f"  [FAIL] Failed:   {total_failed:3d} / {total}")
        print(f"  [WARN] Warnings: {total_warnings:3d} / {total}")

        if total > 0:
            success_rate = (total_passed / total) * 100
            print(f"\n  Success Rate: {success_rate:.1f}%")

        if total_failed == 0:
            print("\n  ALL CRITICAL CHECKS PASSED!")
            print("  System is OPERATIONAL and ready for deployment")
        elif total_failed <= 2:
            print("\n  System is MOSTLY OPERATIONAL with minor issues")
        else:
            print("\n  System has SIGNIFICANT ISSUES that need attention")

        # Print failed items
        if total_failed > 0:
            print("\n  Failed Checks:")
            for item in self.results["failed"]:
                print(f"     - {item}")

        # Print warnings
        if total_warnings > 0:
            print("\n  Warnings:")
            for item in self.results["warnings"][:5]:  # Show first 5
                print(f"     - {item}")
            if len(self.results["warnings"]) > 5:
                print(f"     ... and {len(self.results['warnings']) - 5} more")

        print("\n" + "=" * 80 + "\n")

    def run_full_verification(self):
        """Run complete system verification"""
        self.print_header()

        # Run all checks
        self.verify_dependencies()
        self.verify_core_modules()
        self.verify_next_gen_modules()
        self.verify_api_readiness()
        self.verify_deployment_configs()
        self.run_quick_functionality_test()

        # Print summary
        self.print_summary()

        return len(self.results["failed"]) == 0


if __name__ == "__main__":
    verifier = SystemVerifier()
    success = verifier.run_full_verification()
    sys.exit(0 if success else 1)
