from typing import Any, Dict, Tuple
import logging
import os
import torch
from torch.utils.data import DataLoader
from src.data.loaders import load_csv
from src.models.base_model import BaseModel
from src.utils.logging import setup_logging

class Trainer:
    def __init__(self, model: BaseModel, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]):
        """
        Initializes the Trainer with the model, data loaders, and configuration.

        Parameters:
            model (BaseModel): The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            config (Dict[str, Any]): Configuration settings for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = setup_logging()
        self.epochs = config.get('epochs', 10)
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')

    def train(self) -> None:
        """
        Trains the model using the training data.
        """
        self.model.train()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                self._train_step(batch)
            self.logger.info(f'Epoch {epoch + 1}/{self.epochs} completed.')

            if (epoch + 1) % self.config.get('checkpoint_interval', 1) == 0:
                self.save_checkpoint(epoch)

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Performs a single training step.

        Parameters:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of training data.
        """
        inputs, targets = batch
        self.model.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.model.loss_function(outputs, targets)
        loss.backward()
        self.model.optimizer.step()

    def validate(self) -> None:
        """
        Validates the model using the validation data.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.model.loss_function(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f'Validation loss: {avg_loss}')

    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves the model checkpoint.

        Parameters:
            epoch (int): The current epoch number.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f'Checkpoint saved at {checkpoint_path}')