"""
Numpy-based training script for the Jarvis AI Project.
This version uses only numpy and pandas, avoiding sklearn dependencies.
"""

import argparse
import logging
import sys
import os
from typing import Any, Dict
import yaml

# Add the project root to the Python path to resolve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.numpy_processor import DataProcessor
from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.training.numpy_trainer import NumpyTrainer

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_numpy_training(config_path: str) -> None:
    """
    Loads configuration and executes the full training pipeline using numpy-based components.

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

    try:
        # 2. Load and Prepare Data using DataProcessor
        processor = DataProcessor(
            target_column=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            random_state=config["training"]["seed"]
        )
        
        # Check if data file exists
        data_path = config["data"]["path"]
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            return
        
        # Process data pipeline
        X_train, X_test, y_train, y_test = processor.process_pipeline(data_path)
        logger.info("Data processing completed successfully.")
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # 3. Initialize Model
        model = SimpleNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=config["model"]["hidden_sizes"],
            output_size=config["model"]["output_size"]
        )
        logger.info("Model initialized.")        # 4. Initialize and Run Trainer
        trainer = NumpyTrainer(model=model)
        
        logger.info("Starting model training...")
        metrics = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=config["training"]["epochs"]
        )
        
        # 5. Save components
        model_dir = os.path.dirname(config["training"].get("model_path", "models/"))
        os.makedirs(model_dir, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        processor.save_scaler(preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save model
        model_path = config["training"].get("model_path", "models/trained_model.pkl")
        trainer.save_model(model_path)
        logger.info(f"Trained model saved to {model_path}")
        
        # 6. Print final metrics
        logger.info("Training completed successfully!")
        logger.info("Final Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for the Jarvis project using numpy-only components."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    run_numpy_training(args.config)
