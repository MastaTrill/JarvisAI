"""
Simplified training script for the Jarvis AI Project using scikit-learn.
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

    # Check for required configuration sections
    try:
        data_config = config["data"]
        training_config = config["training"]
        model_config = config["model"]
    except KeyError as e:
        logger.error(f"Missing required configuration section: {e}")
        return

    # 2. Load and Prepare Data using DataProcessor
    try:
        processor = DataProcessor(
            target_column=data_config["target_column"],
            test_size=data_config["test_size"],
            random_state=training_config.get("seed", 42)
        )
    except KeyError as e:
        logger.error(f"Missing required data configuration: {e}")
        return
    
    # Check if data file exists, if not create dummy data
    data_path = data_config["path"]
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found at {data_path}. Creating dummy data.")
        df = processor.create_dummy_data(n_samples=1000, n_features=10)
        # Save dummy data for future use
        if os.path.dirname(data_path):  # Only create directory if path has a directory
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Dummy data saved to {data_path}")
    
    # Process data pipeline
    try:
        X_train, X_test, y_train, y_test = processor.process_pipeline(data_path)
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return

    # Save the fitted preprocessor
    preprocessor_path = training_config["preprocessor_path"]
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    processor.save_scaler(preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    logger.info("Data processing completed successfully.")
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 3. Initialize Model
    model = SimpleNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=model_config["hidden_sizes"],
        output_size=model_config["output_size"],
        config={
            'max_iter': training_config["epochs"],
            'learning_rate_init': training_config["learning_rate"],
            'alpha': model_config.get("alpha", 0.0001)
        }
    )
    logger.info("Model initialized.")

    # 4. Initialize and Run Trainer
    trainer = NumpyTrainer(
        model=model
    )
    
    logger.info("Starting model training...")
    try:
        metrics = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=training_config["epochs"]
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # 5. Save the trained model
    model_path = training_config.get("model_path", "models/trained_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    logger.info(f"Trained model saved to {model_path}")
    
    # 6. Print final metrics
    logger.info("Training completed successfully!")
    logger.info("Final Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")


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
