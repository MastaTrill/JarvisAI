"""
Final training script for Jarvis AI using only numpy
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

    # 2. Load and Prepare Data using DataProcessor
    processor = DataProcessor(
        target_column=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["training"]["seed"]
    )
    
    # Check if data file exists, if not create dummy data
    data_path = config["data"]["path"]
    if not os.path.exists(data_path):
        logger.warning(f"Data file not found. Creating dummy data.")
        df = processor.create_dummy_data(n_samples=1000, n_features=10)
        # Save dummy data for future use
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Dummy data saved to {data_path}")
    
    # Process data pipeline
    X_train, X_test, y_train, y_test = processor.process_pipeline(data_path)

    # Save the fitted preprocessor
    preprocessor_path = config["training"]["preprocessor_path"]
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    processor.save_scaler(preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    logger.info("Data processing completed successfully.")
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Initialize Model
    model = SimpleNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=config["model"]["hidden_sizes"],
        output_size=config["model"]["output_size"],
        config=config["model"]
    )
    logger.info("Model initialized.")

    # 4. Initialize and Run Trainer
    trainer = NumpyTrainer(model=model)
    
    logger.info("Starting model training...")
    metrics = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=config["training"]["epochs"]
    )
    
    # 5. Save the trained model
    model_path = config["training"].get("model_path", "models/trained_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)
    logger.info(f"Trained model saved to {model_path}")
    
    # 6. Print final metrics
    logger.info("Training completed successfully!")
    logger.info("Final Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # 7. Test predictions
    logger.info("Testing predictions on a few samples...")
    test_samples = X_test[:5]
    predictions = trainer.predict(test_samples)
    actual = y_test[:5]
    
    logger.info("Sample Predictions vs Actual:")
    for i, (pred, actual_val) in enumerate(zip(predictions, actual)):
        if len(pred.shape) > 0:
            pred_val = pred[0] if pred.shape[0] > 0 else pred
        else:
            pred_val = pred
        logger.info(f"  Sample {i+1}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}")


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
