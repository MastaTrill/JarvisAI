"""
Test script for Aetheron Platform integration.
Tests WebSocket connection, API endpoints, and real-time training.
"""

import asyncio
import websockets
import requests
import json
import time
from datetime import datetime

class AetheronTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws") + "/ws/test_client"
        
    async def test_websocket_connection(self):
        """Test WebSocket connection."""
        print("Testing WebSocket connection...")
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send ping
                await websocket.send("ping")
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"WebSocket response: {response}")
                return response == "pong"
        except Exception as e:
            print(f"WebSocket test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test basic API endpoints."""
        print("Testing API endpoints...")
        
        # Test root endpoint
        try:
            response = requests.get(f"{self.base_url}/")
            print(f"Root endpoint: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Root endpoint test failed: {e}")
        
        # Test models list endpoint
        try:
            response = requests.get(f"{self.base_url}/api/models/list")
            print(f"Models list endpoint: {response.status_code}")
            if response.status_code == 200:
                models = response.json()
                print(f"Current models: {len(models['models'])}")
        except Exception as e:
            print(f"Models list test failed: {e}")
        
        # Test system metrics endpoint
        try:
            response = requests.get(f"{self.base_url}/api/system/metrics")
            print(f"System metrics endpoint: {response.status_code}")
            if response.status_code == 200:
                metrics = response.json()
                print(f"CPU: {metrics['cpu_usage']}%, Memory: {metrics['memory_usage']}%")
        except Exception as e:
            print(f"System metrics test failed: {e}")
    
    def test_training_api(self):
        """Test training API endpoint."""
        print("Testing training API...")
        
        train_config = {
            "model_name": f"test_model_{int(time.time())}",
            "model_type": "basic",
            "config": {
                "hidden_sizes": [32, 16],
                "epochs": 10,
                "learning_rate": 0.01,
                "batch_size": 32
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/train",
                json=train_config,
                headers={"Content-Type": "application/json"}
            )
            print(f"Training endpoint: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Training started: {result}")
                return result["model_name"]
            else:
                print(f"Training failed: {response.text}")
        except Exception as e:
            print(f"Training test failed: {e}")
        
        return None
    
    def test_data_upload(self):
        """Test data upload functionality."""
        print("Testing data upload...")
        
        # Create sample data file
        sample_data = """feature1,feature2,feature3,target
1.0,2.0,3.0,1
2.0,3.0,4.0,0
3.0,4.0,5.0,1"""
        
        with open("test_data.csv", "w") as f:
            f.write(sample_data)
        
        try:
            with open("test_data.csv", "rb") as f:
                files = {"file": ("test_data.csv", f, "text/csv")}
                response = requests.post(f"{self.base_url}/data/upload", files=files)
                
            print(f"Data upload endpoint: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Upload successful: {result['data_info']['filename']}")
                print(f"Data shape: {result['data_info']['shape']}")
        except Exception as e:
            print(f"Data upload test failed: {e}")
    
    async def listen_for_training_updates(self, duration=30):
        """Listen for training updates via WebSocket."""
        print(f"Listening for training updates for {duration} seconds...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                start_time = time.time()
                while time.time() - start_time < duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "training_update":
                            model_name = data.get("model_name", "unknown")
                            update_data = data.get("data", {})
                            print(f"Training update for {model_name}: "
                                  f"Epoch {update_data.get('epoch', 0)}, "
                                  f"Progress {update_data.get('progress', 0):.1f}%, "
                                  f"Loss {update_data.get('train_loss', 0):.4f}")
                        
                        elif data.get("type") == "training_complete":
                            model_name = data.get("model_name", "unknown")
                            print(f"Training completed for {model_name}")
                            metrics = data.get("data", {}).get("metrics", {})
                            print(f"Final metrics: {metrics}")
                        
                        elif data.get("type") == "system_metrics":
                            metrics = data.get("data", {})
                            print(f"System: CPU {metrics.get('cpu_usage', 0):.1f}%, "
                                  f"Memory {metrics.get('memory_usage', 0):.1f}%")
                        
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send("ping")
                        
        except Exception as e:
            print(f"WebSocket listening failed: {e}")
    
    async def run_full_test(self):
        """Run comprehensive test suite."""
        print("=" * 50)
        print("Aetheron Platform Integration Test")
        print("=" * 50)
        
        # Test WebSocket connection
        ws_success = await self.test_websocket_connection()
        print(f"WebSocket test: {'PASS' if ws_success else 'FAIL'}")
        print()
        
        # Test API endpoints
        self.test_api_endpoints()
        print()
        
        # Test data upload
        self.test_data_upload()
        print()
        
        # Start training
        model_name = self.test_training_api()
        print()
        
        # Listen for updates if training started
        if model_name:
            await self.listen_for_training_updates(20)
        
        print("=" * 50)
        print("Test completed!")
        print("=" * 50)


async def main():
    tester = AetheronTester()
    await tester.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())
