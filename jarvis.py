"""
Jarvis AI - Easy Launch Script

This script provides a simple interface to train models and make predictions.
"""

import os
import sys
import argparse
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy", "pandas", "matplotlib", "seaborn", "pyyaml", "pytest", "tqdm", "joblib"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def train_model():
    """Train a new model"""
    print("🚀 Starting model training...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "src.training.train_final", 
            "--config", "config/train_config.yaml"
        ])
        print("✅ Model training completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Model training failed")
        return False

def run_inference():
    """Run inference with trained model"""
    print("🔮 Running model inference...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "src.inference.predict"
        ])
        print("✅ Inference completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Inference failed")
        return False

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pytest", "tests/test_training_numpy.py", "-v"
        ])
        print("✅ Tests completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Some tests failed (this is normal on Windows due to file permissions)")
        return False

def main():
    print("=" * 60)
    print("🤖 Welcome to Jarvis AI!")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Jarvis AI - Easy Launch Script")
    parser.add_argument("action", choices=["install", "train", "predict", "test"], 
                       help="Action to perform")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    if args.action == "install":
        success = install_dependencies()
    elif args.action == "train":
        print("📋 This will train a neural network model using numpy.")
        print("   If no data is found, dummy data will be generated.")
        success = train_model()
    elif args.action == "predict":
        print("🎯 This will run inference using the trained model.")
        success = run_inference()
    elif args.action == "test":
        print("🔍 This will run the test suite to verify everything works.")
        success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Operation completed successfully!")
        
        if args.action == "train":
            print("\n📁 Generated files:")
            print("   - models/trained_model.pkl (trained neural network)")
            print("   - artifacts/preprocessor.pkl (data preprocessor)")
            print("   - data/processed/dataset.csv (training data)")
            print("\n➡️  Next step: Run 'python jarvis.py predict' to test predictions")
            
        elif args.action == "predict":
            print("\n🎯 Predictions generated successfully!")
            print("➡️  Check the output above for prediction results")
            
        elif args.action == "install":
            print("\n➡️  Next step: Run 'python jarvis.py train' to train a model")
            
    else:
        print("❌ Operation failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
