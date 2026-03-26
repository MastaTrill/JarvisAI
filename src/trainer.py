"""
Trainer class to handle model training, evaluation, and checkpointing.
"""

import logging
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup logger
logger = logging.getLogger(__name__)


class Trainer:
    """
    A class to encapsulate the training and evaluation loop for a PyTorch model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (optim.Optimizer): The optimization algorithm.
            criterion (nn.Module): The loss function.
            device (torch.device): The device to run training on ('cpu' or 'cuda').
            config (Dict[str, Any]): A dictionary containing training configurations.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Trainer initialized for model {model.__class__.__name__} on device {device}")

    def _train_epoch(self, data_loader: DataLoader) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluates the model on a validation or test set."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        The main training loop that iterates over epochs and saves the best model.
        """
        best_val_loss = float('inf')
        epochs = self.config["training"]["epochs"]

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._evaluate(val_loader)

            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Placeholder for experiment tracking (e.g., MLflow or W&B)
            # self.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best_model.pth")
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")

    def _save_checkpoint(self, model_name: str) -> None:
        """Saves the model's state dictionary to the checkpoint directory."""
        try:
            path = os.path.join(self.checkpoint_dir, model_name)
            torch.save(self.model.state_dict(), path)
            logger.debug(f"Checkpoint saved to {path}")
        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")