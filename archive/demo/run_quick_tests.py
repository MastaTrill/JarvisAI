#!/usr/bin/env python3
"""
Lightweight test runner that doesn't load the full API app
Use this instead of pytest for faster feedback
"""

import subprocess
import sys

# Run only tests that don't need the full app
simple_tests = [
    "tests/test_health.py",
    "tests/test_cache.py",
    "tests/test_auth.py",
]

print("Running lightweight tests (no full API load)...")
print("=" * 70)

for test in simple_tests:
    print(f"\n▶ Running {test}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test, "-v", "--tb=short"], cwd="."
    )
    if result.returncode != 0:
        print(f"⚠ {test} failed")

print("\n" + "=" * 70)
print("For full test suite, use: pytest tests/ -v")
print("For single file: pytest tests/test_api_basic.py -v")
