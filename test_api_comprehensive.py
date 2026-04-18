#!/usr/bin/env python3
"""Comprehensive JarvisAI API test script"""

import requests
import json
import time


def test_api_comprehensive():
    base_url = "http://127.0.0.1:8080"

    print("Comprehensive JarvisAI API Testing")
    print("=" * 50)

    # Test 1: Health Check
    try:
        print("\n1. Health Check")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print("   [+] Health check passed")
        else:
            print(f"   [-] Health check failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 2: System Metrics
    try:
        print("\n2. System Metrics")
        response = requests.get(f"{base_url}/api/system/metrics", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   CPU: {data.get('cpu_usage', 'N/A')}%")
            print(f"   Memory: {data.get('memory_usage', 'N/A')}%")
            print("   [+] System metrics retrieved")
        else:
            print(f"   [-] System metrics failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 3: List Models
    try:
        print("\n3. Available Models")
        response = requests.get(f"{base_url}/api/models/list", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"   Found {len(models)} models")
            if models:
                for model in models[:3]:  # Show first 3
                    print(f"     - {model.get('name', 'unnamed')}")
            print("   [+] Models list retrieved")
        else:
            print(f"   [-] Models list failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 4: Agent Tools
    try:
        print("\n4. Agent Tools")
        response = requests.get(f"{base_url}/agent/tools", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            tools = response.json()
            print(f"   Found {len(tools)} tools")
            if tools:
                tool_names = [t.get("name", "unnamed") for t in tools[:5]]
                print(f"     Tools: {', '.join(tool_names)}")
            print("   [+] Agent tools retrieved")
        else:
            print(f"   [-] Agent tools failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 5: Memory Quality
    try:
        print("\n5. Memory Quality")
        response = requests.get(f"{base_url}/agent/memory/quality", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            quality = data.get("overall_quality", "unknown")
            print(f"   Memory Quality: {quality}")
            print("   [+] Memory quality assessed")
        else:
            print(f"   [-] Memory quality failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 6: Quantum Status
    try:
        print("\n6. Quantum Status")
        response = requests.get(f"{base_url}/agent/quantum/status", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            print(f"   Quantum Status: {status}")
            print("   [+] Quantum system operational")
        else:
            print(f"   [-] Quantum status failed: {response.text}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 7: API Documentation
    try:
        print("\n7. API Documentation")
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   [+] Interactive API docs available")
            print(f"   URL: {base_url}/docs")
        else:
            print("   [-] API docs not accessible")
    except Exception as e:
        print(f"   [-] Error: {e}")

    print("\n" + "=" * 50)
    print("API Testing Complete!")
    print(f"Server: {base_url}")
    print("Full API docs: http://localhost:8080/docs")


if __name__ == "__main__":
    test_api_comprehensive()
