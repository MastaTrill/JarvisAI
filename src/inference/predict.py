"""
Simple inference script for Jarvis AI
"""

import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.data.numpy_processor import DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_inference(model_path: str, data_path: str, preprocessor_path: str):
    """
    Run inference on new data
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data for inference
        preprocessor_path: Path to the fitted preprocessor
    """
    logger.info("Starting inference...")
    
    # Load the trained model
    model = SimpleNeuralNetwork(
        input_size=10,  # Will be updated when loading
        hidden_sizes=[64, 32],
        output_size=1
    )
    model.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Load the preprocessor
    processor = DataProcessor()
    processor.load_scaler(preprocessor_path)
    logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    # Load and preprocess the data
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path}")
    else:
        # Generate some sample data for demonstration
        logger.info("Generating sample data for inference...")
        np.random.seed(42)
        sample_data = np.random.randn(5, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        data = pd.DataFrame(sample_data, columns=feature_names)
    
    # Preprocess the data (without target column)
    X_processed = processor.scaler.transform(data.values)
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    # Display results
    logger.info("Inference Results:")
    logger.info("-" * 50)
    for i, pred in enumerate(predictions):
        pred_val = pred[0] if len(pred.shape) > 0 else pred
        logger.info(f"Sample {i+1}: Predicted = {pred_val:.4f}")
    
    logger.info("Inference completed successfully!")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Jarvis model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/trained_model.pkl",
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/inference_data.csv",
        help="Path to the data for inference"
    )
    parser.add_argument(
        "--preprocessor", 
        type=str, 
        default="artifacts/preprocessor.pkl",
        help="Path to the fitted preprocessor"
    )
    
    args = parser.parse_args()
    
    run_inference(args.model, args.data, args.preprocessor)
