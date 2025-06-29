"""
Simplified training utilities for the Jarvis AI Project using scikit-learn.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class SimpleTrainer:
    """
    A simplified trainer for scikit-learn models.
    
    This class handles training, validation, and basic metrics logging.
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        task_type: str = 'regression'
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: scikit-learn model to train
            task_type: 'regression' or 'classification'
        """
        self.model = model
        self.task_type = task_type
        
        # Training history
        self.train_scores = []
        self.val_scores = []
        self.metrics_history = {}
        
        logger.info(f"Trainer initialized for {task_type} task")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs (used for model's max_iter)
            
        Returns:
            Dict[str, float]: Final training metrics
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Update model parameters if it supports max_iter
        if hasattr(self.model, 'set_params'):
            try:
                self.model.set_params(max_iter=epochs)
            except ValueError:
                # Some models might not have max_iter parameter
                pass
        
        # Train the model
        self.model.fit(X_train, y_train.ravel() if len(y_train.shape) > 1 else y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        if self.task_type == 'regression':
            train_score = r2_score(y_train, train_pred)
            val_score = r2_score(y_val, val_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            metrics = {
                'train_r2': train_score,
                'val_r2': val_score,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'final_train_score': train_score,
                'final_val_score': val_score
            }
            
            logger.info(f"Training completed - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
            logger.info(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
            
        else:  # classification
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            
            metrics = {
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                'final_train_score': train_score,
                'final_val_score': val_score
            }
            
            logger.info(f"Training completed - Train Acc: {train_score:.4f}, Val Acc: {val_score:.4f}")
        
        # Store metrics
        self.train_scores.append(metrics['final_train_score'])
        self.val_scores.append(metrics['final_val_score'])
        self.metrics_history = metrics
        
        logger.info("Training process completed")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if hasattr(self.model, 'save'):
            self.model.save(filepath)
        else:
            joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        if hasattr(self.model, 'load'):
            self.model.load(filepath)
        else:
            self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the training history.
        
        Returns:
            Dict[str, Any]: Dictionary containing training metrics history
        """
        return {
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'metrics_history': self.metrics_history
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)


# Alias for backward compatibility
Trainer = SimpleTrainer
