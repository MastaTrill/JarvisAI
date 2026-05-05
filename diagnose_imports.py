#!/usr/bin/env python3
"""Diagnostic script to identify slow imports"""

import sys
import time


def timed_import(module_name):
    """Time an import"""
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        print(f"✓ {module_name:40s} {elapsed:8.3f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {module_name:40s} {elapsed:8.3f}s - ERROR: {e}")
        return False


print("=" * 70)
print("JARVIS AI IMPORT DIAGNOSTICS")
print("=" * 70)

# Core dependencies
modules = [
    "numpy",
    "pandas",
    "fastapi",
    "sqlalchemy",
    "database_models",
    "authentication",
    "database",
    "db_config",
    "admin_dashboard",
]

for mod in modules:
    timed_import(mod)

print("\n" + "=" * 70)
print("Testing main API import (this may take a while)...")
print("=" * 70)
timed_import("api")
