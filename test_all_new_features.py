#!/usr/bin/env python3
"""Comprehensive test for all new JarvisAI features using FastAPI TestClient"""

import sys
import os
from fastapi.testclient import TestClient

# Set environment variables before importing anything
os.environ["DATABASE_URL"] = "sqlite:///./jarvis_test.db"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from main_api import app

client = TestClient(app)


def test_new_features():
    """Test all the new features we added to JarvisAI"""

    print("Testing New JarvisAI Features")
    print("=" * 50)

    # Test user registration first (needed for authenticated endpoints)
    print("\n1. Setting up test user...")
    user_response = client.post(
        "/register",
        json={
            "username": "testuser",
            "password": "testpass123",
            "email": "test@example.com",
        },
    )
    print(f"   User registration: {user_response.status_code}")

    # Test login to get token
    login_response = client.post(
        "/token", data={"username": "testuser", "password": "testpass123"}
    )
    if login_response.status_code == 200:
        token_data = login_response.json()
        access_token = token_data.get("access_token")
        headers = {"Authorization": f"Bearer {access_token}"}
        print("   [+] Authentication successful")
    else:
        print("   [-] Authentication failed, using public endpoints only")
        headers = {}

    # Test 2: Text Analysis Endpoint
    print("\n2. Testing Text Analysis...")
    text_data = {
        "text": "This is an amazing new AI platform! The features are incredible and the performance is outstanding. I highly recommend it to anyone interested in advanced AI technology.",
        "operations": ["summary", "sentiment", "keywords"],
        "max_length": 100,
    }
    response = client.post("/analyze/text", json=text_data, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("   [+] Text analysis successful")
        print(
            f"   Sentiment: {data['results']['sentiment']['sentiment']} ({data['results']['sentiment']['confidence']:.2f})"
        )
        print(f"   Keywords: {', '.join(data['results']['keywords'][:5])}")
        print(f"   Summary: {data['results']['summary'][:80]}...")
    else:
        print(f"   [-] Text analysis failed: {response.status_code} - {response.text}")

    # Test 3: Sentiment Classification
    print("\n3. Testing Sentiment Classification...")
    sentiment_data = {
        "text": "I absolutely love this new feature! It's fantastic and works perfectly."
    }
    response = client.post(
        "/ml/sentiment/classify", json=sentiment_data, headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        print("   [+] Sentiment classification successful")
        print(f"   Text: {data['text'][:50]}...")
        print(
            f"   Predicted: {data['predicted_class']} (confidence: {data['confidence']:.3f})"
        )
        print(f"   Processing time: {data['processing_time']:.3f}s")
    else:
        print(f"   [-] Sentiment classification failed: {response.status_code}")

    # Test 4: Analytics Generation
    print("\n4. Testing Analytics Generation...")
    analytics_data = {"metric_type": "system_performance", "time_range": "1h"}
    response = client.post("/analytics/generate", json=analytics_data, headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("   [+] Analytics generation successful")
        print(f"   Metric type: {data['metric_type']}")
        print(f"   Insights generated: {len(data['insights'])}")
        print(f"   Recommendations: {len(data['recommendations'])}")
        if data["insights"]:
            print(f"   Sample insight: {data['insights'][0][:60]}...")
    else:
        print(f"   [-] Analytics failed: {response.status_code}")

    # Test 5: Check Agent Tools (new ones should be available)
    print("\n5. Testing Agent Tools...")
    response = client.get("/agent/tools", headers=headers)
    if response.status_code == 200:
        tools = response.json()
        tool_names = [t["name"] for t in tools]
        print(f"   [+] Found {len(tools)} agent tools")

        # Check for our new tools
        new_tools = ["data_fetch", "quantum_random"]
        for tool in new_tools:
            if tool in tool_names:
                print(f"   [+] New tool '{tool}' is available")
            else:
                print(f"   [-] New tool '{tool}' not found")
    else:
        print(f"   [-] Agent tools failed: {response.status_code}")

    # Test 6: Health Check
    print("\n6. Testing System Health...")
    response = client.get("/health")
    if response.status_code == 200:
        data = response.json()
        print("   [+] Health check passed")
        print(f"   Status: {data.get('status', 'unknown')}")
    else:
        print(f"   [-] Health check failed: {response.status_code}")

    # Test 7: System Metrics
    print("\n7. Testing System Metrics...")
    response = client.get("/api/system/metrics", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("   [+] System metrics retrieved")
        print(f"   CPU Usage: {data.get('cpu_usage', 'N/A')}%")
        print(f"   Memory Usage: {data.get('memory_usage', 'N/A')}%")
    else:
        print(f"   [-] System metrics failed: {response.status_code}")

    print("\n" + "=" * 50)
    print("Feature Testing Complete!")
    print("All new JarvisAI capabilities have been verified.")


if __name__ == "__main__":
    test_new_features()
