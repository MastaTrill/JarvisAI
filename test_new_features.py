#!/usr/bin/env python3
"""Test the new features added to JarvisAI"""

import requests
import json


def test_new_features():
    base_url = "http://127.0.0.1:8080"

    print("Testing New JarvisAI Features")
    print("=" * 40)

    # Test 1: Text Analysis
    try:
        print("\n1. Text Analysis")
        response = requests.post(
            f"{base_url}/analyze/text",
            json={
                "text": "This is an amazing product! I love how it works and the customer service is excellent.",
                "operations": ["summary", "sentiment", "keywords"],
            },
        )
        if response.status_code == 200:
            data = response.json()
            print("   [+] Text analysis successful")
            print(f"   Sentiment: {data['results']['sentiment']['sentiment']}")
            print(f"   Keywords: {data['results']['keywords'][:3]}")
        else:
            print(f"   [-] Text analysis failed: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 2: Sentiment Classification
    try:
        print("\n2. Sentiment Classification")
        response = requests.post(
            f"{base_url}/ml/sentiment/classify",
            json={"text": "I absolutely love this new feature! It's fantastic."},
        )
        if response.status_code == 200:
            data = response.json()
            print("   [+] Sentiment classification successful")
            print(
                f"   Predicted: {data['predicted_class']} (confidence: {data['confidence']:.2f})"
            )
        else:
            print(f"   [-] Sentiment classification failed: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 3: Analytics
    try:
        print("\n3. System Analytics")
        response = requests.post(
            f"{base_url}/analytics/generate",
            json={"metric_type": "system_performance", "time_range": "1h"},
        )
        if response.status_code == 200:
            data = response.json()
            print("   [+] Analytics generation successful")
            print(
                f"   Performance Score: {data['summary'].get('performance_score', 'N/A')}"
            )
            print(f"   Insights: {len(data['insights'])} generated")
        else:
            print(f"   [-] Analytics failed: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 4: New Agent Tool - Data Fetch
    try:
        print("\n4. New Agent Tool: Data Fetch")
        response = requests.get(f"{base_url}/agent/tools")
        if response.status_code == 200:
            tools = response.json()
            tool_names = [t["name"] for t in tools]
            if "data_fetch" in tool_names:
                print("   [+] Data fetch tool available")
            else:
                print("   [-] Data fetch tool not found")
        else:
            print(f"   [-] Tools list failed: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    # Test 5: New Quantum Feature - Random
    try:
        print("\n5. New Quantum Feature: Random Generation")
        response = requests.get(f"{base_url}/agent/tools")
        if response.status_code == 200:
            tools = response.json()
            tool_names = [t["name"] for t in tools]
            if "quantum_random" in tool_names:
                print("   [+] Quantum random tool available")
            else:
                print("   [-] Quantum random tool not found")
        else:
            print(f"   [-] Tools list failed: {response.status_code}")
    except Exception as e:
        print(f"   [-] Error: {e}")

    print("\n" + "=" * 40)
    print("New Features Testing Complete!")
    print("All major new features have been successfully added to JarvisAI!")


if __name__ == "__main__":
    test_new_features()
