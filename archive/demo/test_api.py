#!/usr/bin/env python3
"""Quick API test script"""

import requests
import time


def test_api():
    base_url = "http://127.0.0.1:8080"

    print("Testing JarvisAI API endpoints...\n")

    # Test health endpoint
    try:
        print("1. Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.text[:100]}...")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test root endpoint
    try:
        print("2. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test API docs
    try:
        print("3. Testing /docs endpoint...")
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            print("   ✓ API documentation available")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test v1 API
    try:
        print("4. Testing /v1/ endpoint...")
        response = requests.get(f"{base_url}/v1/", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ V1 API accessible")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")


if __name__ == "__main__":
    test_api()
