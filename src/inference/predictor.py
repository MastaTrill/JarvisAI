"""
Model prediction utilities for Jarvis AI.

This module provides prediction capabilities for various ML models
and handles input preprocessing and output postprocessing.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Any
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    A generic model predictor that can handle different types of models.
    
    Supports scikit-learn models, PyTorch models, and can be extended for other frameworks.
    """
    
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor file
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.is_loaded = False
        self.preprocessor_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Try loading as PyTorch model first
            if path.endswith('.pth') or path.endswith('.pt'):
                self.model = torch.load(path, map_location=self.device)
                if isinstance(self.model, dict):
                    # Handle state dict format
                    # This requires knowing the model architecture
                    logger.warning("Loaded state dict format. Model architecture needed.")
                elif isinstance(self.model, nn.Module):
                    self.model.to(self.device)
                    self.model.eval()
            else:
                # Try loading as scikit-learn model
                self.model = joblib.load(path)
                
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: Optional[str] = None) -> None:
        """
        Load a preprocessor from disk.
        
        Args:
            preprocessor_path: Path to the preprocessor file. If None, uses self.preprocessor_path
        """
        path = preprocessor_path or self.preprocessor_path
        if not path:
            logger.warning("No preprocessor path provided. Predictions will use raw data.")
            return
            
        if not Path(path).exists():
            logger.warning(f"Preprocessor file not found: {path}")
            return
        
        try:
            self.preprocessor = joblib.load(path)
            self.preprocessor_loaded = True
            logger.info(f"Preprocessor loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor from {path}: {e}")
            raise
    
    def preprocess_input(self, data: Union[np.ndarray, pd.DataFrame, list]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data to preprocess
            
        Returns:
            Preprocessed data as numpy array
        """
        # Convert to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Apply preprocessor if available
        if self.preprocessor_loaded and self.preprocessor is not None:
            try:
                data = self.preprocessor.transform(data)
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}. Using raw data.")
        
        return data
    
    def predict(self, data: Union[np.ndarray, pd.DataFrame, list]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions as numpy array
            
        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        try:
            # Handle PyTorch models
            if isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(processed_data).to(self.device)
                    predictions = self.model(input_tensor)
                    return predictions.cpu().numpy()
            
            # Handle scikit-learn models
            else:
                predictions = self.model.predict(processed_data)
                return predictions
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_proba(self, data: Union[np.ndarray, pd.DataFrame, list]) -> np.ndarray:
        """
        Make probability predictions on input data (for classification models).
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction probabilities as numpy array
            
        Raises:
            ValueError: If model is not loaded or doesn't support probability prediction
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        processed_data = self.preprocess_input(data)
        
        try:
            # Handle PyTorch models
            if isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(processed_data).to(self.device)
                    predictions = self.model(input_tensor)
                    # Apply softmax for multi-class classification
                    if predictions.shape[1] > 1:
                        predictions = torch.softmax(predictions, dim=1)
                    return predictions.cpu().numpy()
            
            # Handle scikit-learn models
            else:
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(processed_data)
                    return predictions
                else:
                    raise ValueError("Model does not support probability prediction")
                    
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise
    
    def batch_predict(self, data: Union[np.ndarray, pd.DataFrame, list], batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on large datasets in batches.
        
        Args:
            data: Input data for prediction
            batch_size: Size of each batch
            
        Returns:
            All predictions concatenated as numpy array
        """
        processed_data = self.preprocess_input(data)
        
        if len(processed_data) <= batch_size:
            return self.predict(processed_data)
        
        predictions = []
        for i in range(0, len(processed_data), batch_size):
            batch = processed_data[i:i + batch_size]
            batch_pred = self.predict(batch)
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)


def load_model_and_predict(
    model_path: str,
    data: Union[np.ndarray, pd.DataFrame, list],
    preprocessor_path: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to load a model and make predictions in one call.
    
    Args:
        model_path: Path to the saved model
        data: Input data for prediction
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        Model predictions
    """
    predictor = ModelPredictor(model_path, preprocessor_path)
    predictor.load_model()
    if preprocessor_path:
        predictor.load_preprocessor()
    
    return predictor.predict(data)
