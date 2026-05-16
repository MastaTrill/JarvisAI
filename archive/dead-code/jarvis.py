"""
Jarvis AI - Command Line Interface

This script provides a simple interface to install dependencies, train models,
run predictions, execute tests, manage Docker, deploy, and more.
"""

import argparse
import os
import subprocess
import sys
import json


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
    """Run the test suite."""
    print("[test] Running tests...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"]
        )
        print("[ok] Tests completed.")
        return True
    except subprocess.CalledProcessError:
        print("[warn] Some tests failed.")
        return False


def run_server() -> bool:
    """Start the API server."""
    print("[server] Starting Jarvis AI server...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main_api:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to start server.")
        return False


def run_verification() -> bool:
    """Run system verification."""
    print("[verify] Running system verification...")
    try:
        subprocess.check_call([sys.executable, "verify_system_status.py"])
        print("[ok] Verification completed.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Verification failed.")
        return False


def clear_cache() -> bool:
    """Clear the LLM cache."""
    print("[cache] Clearing cache...")
    try:
        from cache import llm_cache

        llm_cache.clear()
        print("[ok] Cache cleared.")
        return True
    except Exception as e:
        print(f"[error] Failed to clear cache: {e}")
        return False


def docker_build() -> bool:
    """Build Docker image."""
    print("[docker] Building Docker image...")
    try:
        subprocess.check_call(["docker", "build", "-t", "jarvis-ai:latest", "."])
        print("[ok] Docker image built successfully.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Docker build failed. Is Docker installed?")
        return False


def docker_run() -> bool:
    """Run Docker container."""
    print("[docker] Starting Docker container...")
    try:
        subprocess.check_call(
            [
                "docker",
                "run",
                "-d",
                "-p",
                "8000:8000",
                "--name",
                "jarvis-ai",
                "jarvis-ai:latest",
            ]
        )
        print("[ok] Docker container running at http://localhost:8000")
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to run container.")
        return False


def docker_stop() -> bool:
    """Stop Docker container."""
    print("[docker] Stopping Docker container...")
    try:
        subprocess.check_call(["docker", "stop", "jarvis-ai"])
        print("[ok] Container stopped.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to stop container.")
        return False


def docker_logs() -> bool:
    """Show Docker container logs."""
    print("[docker] Showing container logs...")
    try:
        subprocess.check_call(["docker", "logs", "-f", "jarvis-ai"])
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to show logs.")
        return False


def deploy_local() -> bool:
    """Deploy locally using docker-compose."""
    print("[deploy] Deploying locally with docker-compose...")
    try:
        subprocess.check_call(["docker-compose", "up", "-d"])
        print("[ok] Deployed locally. Access at http://localhost:8000")
        return True
    except subprocess.CalledProcessError:
        print("[error] Deployment failed.")
        return False


def deploy_azure() -> bool:
    """Deploy to Azure."""
    print("[deploy] Deploying to Azure...")
    print("[info] This requires Azure CLI and proper credentials.")
    try:
        subprocess.check_call(
            [
                "az",
                "deployment",
                "group",
                "create",
                "--resource-group",
                "jarvis-rg",
                "--template-file",
                "azuredeploy.json",
            ]
        )
        print("[ok] Azure deployment initiated.")
        return True
    except subprocess.CalledProcessError:
        print("[error] Azure deployment failed. Is Azure CLI installed?")
        return False


def ssh_connect(host: str, user: str) -> bool:
    """SSH connection info (opens terminal)."""
    print(f"[ssh] To connect to {user}@{host}, run:")
    print(f"   ssh {user}@{host}")
    print("[info] Make sure you have SSH keys configured.")
    return True


def run_celery_worker() -> bool:
    """Start Celery worker for async tasks."""
    print("[celery] Starting Celery worker...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "celery_app",
                "worker",
                "--loglevel=info",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to start Celery worker.")
        return False


def run_celery_beat() -> bool:
    """Start Celery beat scheduler."""
    print("[celery] Starting Celery beat scheduler...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "celery_app",
                "beat",
                "--loglevel=info",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        print("[error] Failed to start Celery beat.")
        return False


def list_tasks() -> bool:
    """List available async tasks."""
    print("[celery] Available tasks:")
    print("   - training.train_model (async model training)")
    print("   - inference.run_prediction (async inference)")
    print("   - data.process_dataset (async data processing)")
    return True


def show_cache_stats() -> bool:
    """Show cache statistics."""
    print("[cache] Cache Statistics:")
    try:
        from cache import llm_cache

        print(
            f"   Type: {'Redis' if llm_cache.use_redis and llm_cache._redis else 'Memory'}"
        )
        print(f"   Entries: {llm_cache.memory_cache.size()}")
        return True
    except Exception as e:
        print(f"[error] {e}")
        return False


def main() -> None:
    """Main entry point for Jarvis AI."""
    print("=" * 60)
    print("Welcome to Jarvis AI!")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Jarvis AI - Command Line Interface")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Core commands
    subparsers.add_parser("install", help="Install dependencies")
    subparsers.add_parser("train", help="Train a neural network model")
    subparsers.add_parser("predict", help="Run inference with trained model")
    subparsers.add_parser("test", help="Run the test suite")
    subparsers.add_parser("server", help="Start the API server")
    subparsers.add_parser("verify", help="Run system verification")
    subparsers.add_parser("cache-clear", help="Clear LLM response cache")
    subparsers.add_parser("cache-stats", help="Show cache statistics")

    # Docker commands
    docker_parser = subparsers.add_parser("docker", help="Docker management")
    docker_parser.add_argument(
        "subaction", choices=["build", "run", "stop", "logs"], help="Docker action"
    )

    # Deployment commands
    deploy_parser = subparsers.add_parser("deploy", help="Deployment commands")
    deploy_parser.add_argument(
        "target", choices=["local", "azure"], help="Deployment target"
    )

    # SSH command
    ssh_parser = subparsers.add_parser("ssh", help="SSH connection info")
    ssh_parser.add_argument("--host", default="localhost", help="SSH host")
    ssh_parser.add_argument("--user", default="root", help="SSH user")

    # Celery commands
    celery_parser = subparsers.add_parser("celery", help="Celery async tasks")
    celery_parser.add_argument(
        "subaction", choices=["worker", "beat", "list"], help="Celery action"
    )

    args = parser.parse_args()

    if args.action == "help" or args.action is None:
        print("""
 Jarvis AI Commands:
   Core:
     install       - Install dependencies (CPU-only PyTorch)
     train         - Train a neural network model
     predict       - Run inference with trained model
     test          - Run the test suite
     server        - Start the API server
     verify        - Run system verification
     cache-clear   - Clear LLM response cache
     cache-stats   - Show cache statistics

   Docker:
     docker build  - Build Docker image
     docker run    - Run Docker container
     docker stop   - Stop Docker container
     docker logs   - Show container logs

   Deployment:
     deploy local - Deploy with docker-compose
     deploy azure - Deploy to Azure

   SSH:
     ssh --host <host> --user <user> - Show SSH connection info

   Celery:
     celery worker - Start async worker
     celery beat   - Start beat scheduler
     celery list   - List available tasks
        """)
        return

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
    elif args.action == "server":
        success = run_server()
    elif args.action == "verify":
        success = run_verification()
    elif args.action == "cache-clear":
        success = clear_cache()
    elif args.action == "cache-stats":
        success = show_cache_stats()
    elif args.action == "docker":
        if args.subaction == "build":
            success = docker_build()
        elif args.subaction == "run":
            success = docker_run()
        elif args.subaction == "stop":
            success = docker_stop()
        elif args.subaction == "logs":
            success = docker_logs()
    elif args.action == "deploy":
        if args.target == "local":
            success = deploy_local()
        elif args.target == "azure":
            success = deploy_azure()
    elif args.action == "ssh":
        success = ssh_connect(args.host, args.user)
    elif args.action == "celery":
        if args.subaction == "worker":
            success = run_celery_worker()
        elif args.subaction == "beat":
            success = run_celery_beat()
        elif args.subaction == "list":
            success = list_tasks()

    print("\n" + "=" * 60)
    if success:
        print("[ok] Operation completed successfully.")
        if args.action == "train":
            print("\n[files] Generated files:")
            print("   - models/trained_model.pkl")
            print("   - data/processed/dataset.csv")
        elif args.action == "server":
            print("\n[info] Server running at http://localhost:8000")
            print("   - API docs: http://localhost:8000/docs")
            print("   - Dashboard: http://localhost:8000/")
    else:
        print("[error] Operation failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
