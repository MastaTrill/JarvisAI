"""
Model prediction utilities for Jarvis AI.

This module provides prediction capabilities for various ML models
and handles input preprocessing and output postprocessing.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional
import joblib

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    A generic model predictor that can handle different types of models.
    
    Supports scikit-learn models, PyTorch models, and can be extended for other frameworks.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.is_loaded = False
        self.preprocessor_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the model file. If None, uses self.model_path
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        path = model_path or self.model_path
        if not path:
            raise ValueError("No model path provided")
            
        model_file = Path(path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            # Check file extension to determine loading method
            if Path(path).suffix == '.pth':
                # PyTorch model
                import torch
                self.model = torch.load(path, map_location='cpu')
                # If it's a state dict, we need to load it into a model
                if isinstance(self.model, dict) and 'state_dict' in self.model:
                    # This is a checkpoint with state_dict, need model architecture
                    logger.warning("Loaded state dict, but no model architecture available")
                    self.model = self.model['state_dict']
                elif hasattr(self.model, 'eval'):
                    # This is a complete model
                    self.model.eval()
            else:
                # Try loading as joblib file (common for scikit-learn)
                self.model = joblib.load(path)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise

    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load a fitted preprocessor from disk
        
        Args:
            preprocessor_path: Path to the saved preprocessor file
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.preprocessor_loaded = True
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise

    def predict(self, inputs: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            inputs: Input data for prediction
            
        Returns:
            Model predictions
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Convert inputs to numpy array if needed
        if isinstance(inputs, list):
            inputs = np.array(inputs)
            
        # Apply preprocessing if available
        if self.preprocessor_loaded and self.preprocessor is not None:
            inputs = self.preprocessor.transform(inputs)
        
        try:
            # Check if this is a PyTorch model
            if hasattr(self.model, 'forward') or hasattr(self.model, '__call__'):
                # PyTorch model
                import torch
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.FloatTensor(inputs)
                
                with torch.no_grad():
                    self.model.eval()
                    predictions = self.model(inputs)
                    
                # Convert back to numpy
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.numpy()
                    
                return predictions.flatten()
            else:
                # Scikit-learn or other model
                predictions = self.model.predict(inputs)
                return predictions                
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise


class DataPreprocessor:
    """
    Data preprocessing utilities for model inference
    """

    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = None
        self.is_fitted = False

    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load a fitted preprocessor from disk
        
        Args:
            preprocessor_path: Path to the saved preprocessor file
        """
        try:
            self.scaler = joblib.load(preprocessor_path)
            self.is_fitted = True
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using the loaded preprocessor
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call load_preprocessor() first.")
        
        return self.scaler.transform(data)


def run_demo():
    """Demonstration of the inference pipeline"""
    print("=== Jarvis AI Inference Demo ===")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    
    # Create and fit a simple preprocessor
    from sklearn.preprocessing import StandardScaler
    preprocessor = StandardScaler()
    X_scaled = preprocessor.fit_transform(X)
    
    # Save the preprocessor
    Path("../models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "../models/demo_preprocessor.pkl")
    print("✓ Sample preprocessor saved")
    
    # Create sample predictions (since we don't have a real model yet)
    sample_predictions = np.random.randn(5)
    
    print("✓ Inference demo completed")
    print(f"✓ Sample predictions shape: {sample_predictions.shape}")
    
    return sample_predictions


if __name__ == "__main__":
    run_demo()


class DataPreprocessor:
    """
    Handle data preprocessing for model inference.
    """
    
    @staticmethod
    def normalize_features(data: np.ndarray) -> np.ndarray:
        """
        Normalize features to 0-1 range.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        
        normalized = (data - data_min) / data_range
        logger.debug("Features normalized to [0, 1] range")
        return normalized
    
    @staticmethod
    def standardize_features(data: np.ndarray) -> np.ndarray:
        """
        Standardize features to have mean=0 and std=1.
        
        Args:
            data: Input data
            
        Returns:
            Standardized data
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1
        
        standardized = (data - mean) / std
        logger.debug("Features standardized (mean=0, std=1)")
        return standardized


def create_sample_data(
    n_samples: int = 100, n_features: int = 4
) -> np.ndarray:
    """
    Create sample data for testing predictions.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        
    Returns:
        Generated sample data
    """
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(n_samples, n_features)
    logger.info(f"Created sample data with shape: {data.shape}")
    return data


def demo_prediction_pipeline():
    """
    Demonstrate a complete prediction pipeline.
    """
    logger.info("Starting prediction pipeline demo")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
      # Preprocess data
    normalized_data = preprocessor.normalize_features(sample_data)
    standardized_data = preprocessor.standardize_features(sample_data)
    
    logger.info("Sample preprocessing completed")
    logger.info(f"Original data shape: {sample_data.shape}")
    logger.info(
        f"Normalized data range: [{normalized_data.min():.3f}, "
        f"{normalized_data.max():.3f}]"
    )
    logger.info(
        f"Standardized data mean: {standardized_data.mean():.3f}, "
        f"std: {standardized_data.std():.3f}"
    )
    
    # Note: Actual model prediction would require a trained model
    logger.info("To use ModelPredictor, load a trained model first:")
    logger.info("predictor = ModelPredictor('path/to/model.pkl')")
    logger.info("predictor.load_model()")
    logger.info("predictions = predictor.predict(preprocessed_data)")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_prediction_pipeline()
