"""
Simple neural network implementation using scikit-learn for compatibility
"""

import logging
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator
import joblib
from typing import Optional, Union

logger = logging.getLogger(__name__)


class SimpleNeuralNetwork(BaseEstimator):
    """
    Simple neural network wrapper using scikit-learn's MLP
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        config: Optional[dict] = None,
        task_type: str = 'regression'
    ):
        """
        Initialize neural network

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            config: Optional dictionary with configuration
            task_type: 'regression' or 'classification'
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.config = config or {}
        self.task_type = task_type
        self.is_trained = False
        
        # Create the appropriate model
        if task_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_sizes),
                activation=self.config.get('activation', 'relu'),
                alpha=self.config.get('alpha', 0.0001),
                max_iter=self.config.get('max_iter', 200),
                random_state=42
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_sizes),
                activation=self.config.get('activation', 'relu'),
                alpha=self.config.get('alpha', 0.0001),
                max_iter=self.config.get('max_iter', 200),
                random_state=42
            )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ) -> None:
        """
        Train the neural network

        Args:
            x: Training features
            y: Training targets
            epochs: Number of training epochs (mapped to max_iter)
            learning_rate: Learning rate (mapped to learning_rate_init)
            batch_size: Batch size (not used in scikit-learn MLP)
        """
        # Update model parameters
        self.model.set_params(
            max_iter=epochs,
            learning_rate_init=learning_rate
        )
        
        logger.info(f"Training neural network with {epochs} epochs")
        self.model.fit(x, y.ravel() if len(y.shape) > 1 else y)
        self.is_trained = True
        logger.info("Training completed")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            x: Input features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(x)
        return predictions.reshape(-1, 1) if len(predictions.shape) == 1 else predictions

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Make probability predictions (for classification only)

        Args:
            x: Input features

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        return self.model.predict_proba(x)

    def save(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'model': self.model,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'task_type': self.task_type,
            'config': self.config,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.input_size = model_data['input_size']
        self.hidden_sizes = model_data['hidden_sizes']
        self.output_size = model_data['output_size']
        self.task_type = model_data['task_type']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {filepath}")

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Return the score of the model

        Args:
            x: Test features
            y: Test targets

        Returns:
            Model score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        return self.model.score(x, y.ravel() if len(y.shape) > 1 else y)
