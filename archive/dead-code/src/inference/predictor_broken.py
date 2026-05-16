"""
Model prediction and inference utilities for Jarvis AI
"""

import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Union, Optional, Any
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    A unified predictor class that can handle different types of models
    and preprocessing pipelines.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = None
        self.model_path = model_path
        self.is_loaded = False

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
        
        # Convert input to numpy array if needed
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        
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
            logger.error(f"Prediction failed: {e}")
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


# Combined predictor that includes preprocessing
class ModelPredictor:
    """
    Enhanced model predictor with preprocessing capabilities
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor with optional model path"""
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.is_loaded = False
        self.preprocessor_loaded = False
    
    def load_model(self, model_path: str) -> None:
        """Load model from file"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            if path.suffix == '.pth':
                import torch
                self.model = torch.load(path, map_location='cpu')
                if hasattr(self.model, 'eval'):
                    self.model.eval()
            else:
                self.model = joblib.load(path)
                
            self.is_loaded = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load preprocessor from file"""
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.preprocessor_loaded = True
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    def predict(self, inputs: Union[np.ndarray, list]) -> np.ndarray:
        """Make predictions with optional preprocessing"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
            
        # Convert to numpy array
        if isinstance(inputs, list):
            inputs = np.array(inputs)
            
        # Apply preprocessing if available
        if self.preprocessor_loaded and self.preprocessor is not None:
            inputs = self.preprocessor.transform(inputs)
        
        # Make prediction
        try:
            if hasattr(self.model, 'forward') or hasattr(self.model, '__call__'):
                # PyTorch model
                import torch
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.FloatTensor(inputs)
                
                with torch.no_grad():
                    predictions = self.model(inputs)
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.numpy()
                return predictions.flatten()
            else:
                # Scikit-learn model
                return self.model.predict(inputs)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


def run_demo():
    """Demonstration of the inference pipeline"""
    print("=== Jarvis AI Inference Demo ===")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    
    # Create and fit a simple preprocessor
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
