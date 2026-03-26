"""
Training utilities for the Jarvis AI Project.

This module contains the Trainer class for handling the training loop,
validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os
import mlflow

logger = logging.getLogger(__name__)


class Trainer:
    """
    A comprehensive trainer for PyTorch models with MLflow integration.
    
    This class handles the complete training workflow including:
    - Training and validation loops
    - Loss computation and backpropagation
    - Metrics logging with MLflow
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to run training on (CPU/GPU)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.debug(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple[float, float]: Average validation loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy (for classification tasks)
                if len(output.shape) > 1 and output.shape[1] > 1:
                    # Multi-class classification
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                else:
                    # Regression or binary classification
                    # For regression, we'll use R² score approximation
                    total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Dict[str, float]: Final training metrics
        """
        best_val_loss = float('inf')
        
        # Create checkpoint directory if specified
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_scores.append(val_accuracy)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            # Print progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss and checkpoint_dir:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
        
        # Return final metrics
        final_metrics = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_accuracy': self.val_scores[-1],
            'best_val_loss': best_val_loss
        }
        
        logger.info("Training completed")
        return final_metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get the training history.
        
        Returns:
            Dict[str, list]: Dictionary containing training metrics history
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores
        }
