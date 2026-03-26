"""
Jarvis AI - Easy Launch Script

This script provides a simple interface to install dependencies, train models,
run predictions, and execute the lightweight test suite.
"""

import argparse
import os
import subprocess
import sys


def install_dependencies() -> bool:
    """Install the baseline dependencies needed for local development."""
    print("[setup] Installing dependencies...")
    try:
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "pyyaml",
                "pytest",
                "tqdm",
                "joblib",
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.5.1+cpu",
                "torchvision==0.20.1+cpu",
                "torchaudio==2.5.1+cpu",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        print("[ok] Dependencies (including CPU-only PyTorch) installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to install dependencies.")
        return False


def train_model() -> bool:
    """Train a new model."""
    print("[train] Starting model training...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "src.training.train_final",
                "--config",
                "config/train_config.yaml",
            ]
        )
        print("[ok] Model training completed.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Model training failed.")
        return False


def run_inference() -> bool:
    """Run inference with a trained model."""
    print("[predict] Running model inference...")
    try:
        subprocess.check_call([sys.executable, "-m", "src.inference.predict"])
        print("[ok] Inference completed.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Inference failed.")
        return False


def run_tests() -> bool:
    """Run the lightweight test suite."""
    print("[test] Running tests...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pytest", "tests/test_training_numpy.py", "-v"]
        )
        print("[ok] Tests completed.")
        return True
    except subprocess.CalledProcessError:
        print("[warn] Some tests failed (this can happen on Windows due to file permissions).")
        return False


def main() -> None:
    """Main entry point for Jarvis AI."""
    print("=" * 60)
    print("Welcome to Jarvis AI!")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Jarvis AI - Easy Launch Script")
    parser.add_argument(
        "action",
        choices=["install", "train", "predict", "test"],
        help="Action to perform",
    )

    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    success = False
    if args.action == "install":
        success = install_dependencies()
    elif args.action == "train":
        print("[info] This will train a neural network model using numpy.")
        print("   If no data is found, dummy data will be generated.")
        success = train_model()
    elif args.action == "predict":
        print("[info] This will run inference using the trained model.")
        success = run_inference()
    elif args.action == "test":
        print("[info] This will run the test suite to verify everything works.")
        success = run_tests()

    print("\n" + "=" * 60)
    if success:
        print("[ok] Operation completed successfully.")
        if args.action == "train":
            print("\n[files] Generated files:")
            print("   - models/trained_model.pkl (trained neural network)")
            print("   - data/processed/dataset.csv (training data)")
            print("\n[next] Run 'python jarvis.py predict' to test predictions.")
        elif args.action == "predict":
            print("\n[ok] Predictions generated successfully.")
            print("[next] Check the output above for prediction results.")
        elif args.action == "install":
            print("\n[next] Run 'python jarvis.py train' to train a model.")
    else:
        print("[error] Operation failed. Please check the error messages above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
