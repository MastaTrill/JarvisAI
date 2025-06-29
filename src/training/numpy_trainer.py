"""
Simple trainer for numpy-based neural networks
"""

import logging
import numpy as np
from typing import Dict
import pickle

logger = logging.getLogger(__name__)


class NumpyTrainer:
    """
    Simple trainer for numpy-based neural networks
    """
    
    def __init__(self, model):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model to train
        """
        self.model = model
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        
        logger.info("Trainer initialized")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100
    ) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            
        Returns:
            Dictionary of final metrics
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs)
        
        # Calculate final metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        # Calculate losses (MSE)
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
            
        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val) ** 2)
        
        # Store metrics
        self.train_scores.append(train_score)
        self.val_scores.append(val_score)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        metrics = {
            'final_train_score': train_score,
            'final_val_score': val_score,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        logger.info(f"Training completed - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
        logger.info(f"Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        self.model.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_training_history(self) -> Dict[str, list]:
        """Get training history"""
        return {
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# Alias for compatibility
Trainer = NumpyTrainer
