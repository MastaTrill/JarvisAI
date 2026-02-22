"""
Main training script for the Jarvis AI Project.

This script orchestrates the entire training process by:
1. Loading configuration from a YAML file.
2. Setting up an MLflow experiment for tracking.
3. Loading and preparing the data.
4. Initializing the model, optimizer, and loss function.
5. Running the training and validation loops using the Trainer class.
"""

import argparse
import logging
import sys
import os
from typing import Any, Dict

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the Python path to resolve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.processor import DataProcessor
from src.models.neural_network import SimpleNeuralNetwork
from src.trainer import Trainer

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_training(config_path: str) -> None:
    """
    Loads configuration and executes the full training pipeline.

    Args:
        config_path (str): Path to the training configuration YAML file.
    """
    # 1. Load Configuration
    try:
        with open(config_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Failed to load or parse configuration file: {e}")
        return

    # 2. Set up MLflow Experiment
    mlflow.set_experiment(config["mlflow_config"]["experiment_name"])

    with mlflow.start_run():
        # Log parameters from the config file
        logger.info("Logging parameters to MLflow.")
        mlflow.log_params(config["data"])
        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])

        # 3. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # 4. Load and Prepare Data

        processor = DataProcessor(
            target_column=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            random_state=config["training"]["seed"],
        )
        X_train, X_test, y_train, y_test = processor.process_pipeline(
            config["data"]["path"]
        )

        # Convert to PyTorch Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=config["training"]["batch_size"],
        )
        logger.info("Data loaders created successfully.")

        # 5. Initialize Model, Optimizer, and Criterion
        model = SimpleNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=config["model"]["hidden_sizes"],
            output_size=config["model"]["output_size"],
            config={
                "activation": config["model"].get("activation", "relu"),
                "dropout": config["model"].get("dropout", 0.2),
            },
        )
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
        criterion = nn.MSELoss()
        logger.info("Model, optimizer, and criterion initialized.")

        # 6. Initialize and Run Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
        )
        logger.info("Starting model training...")
        trainer.train(train_loader=train_loader, val_loader=val_loader)
        logger.info("Training process completed and logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for the Jarvis project."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    run_training(args.config)
