from typing import Any, Dict
import logging
import os
import mlflow
import torch
from torch.utils.data import DataLoader
from src.data.loaders import DataLoader as CustomDataLoader
from src.models.base_model import BaseModel

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
        self.logger = self.setup_logging()
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_logging(self) -> logging.Logger:
        """
        Sets up the logging configuration.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger

    def train(self, epochs: int):
        """
        Trains the model for a specified number of epochs.

        Parameters:
            epochs (int): Number of epochs to train the model.
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                loss = self.model.training_step(batch)
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
            self.validate(epoch)

            # Save checkpoint
            self.save_checkpoint(epoch)

    def validate(self, epoch: int):
        """
        Validates the model on the validation dataset.

        Parameters:
            epoch (int): Current epoch number.
        """
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                val_loss = self.model.validation_step(batch)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f}")

    def save_checkpoint(self, epoch: int):
        """
        Saves the model checkpoint.

        Parameters:
            epoch (int): Current epoch number.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def log_experiment(self):
        """
        Logs the experiment details to MLflow.
        """
        mlflow.start_run()
        mlflow.log_params(self.config)
        mlflow.log_artifacts(self.checkpoint_dir)
        mlflow.end_run()