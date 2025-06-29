"""
Simple neural network implementation using PyTorch
"""

import logging
from abc import ABC
from typing import Optional

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for neural_network.py. Please install it with 'pip install numpy'.")

try:
    import torch
    from torch import nn, optim
except ImportError:
    raise ImportError("PyTorch is required for neural_network.py. Please install it with 'pip install torch'.")

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base model interface for consistency"""

    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.config = config or {}
        self.is_trained = False


class SimpleNeuralNetwork(BaseModel, nn.Module):
    """
    Simple feedforward neural network for classification or regression
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        config: Optional[dict] = None
    ):
        """
        Initialize neural network

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            config: Optional dictionary with keys 'activation' and 'dropout'
        """
        BaseModel.__init__(self, name="SimpleNeuralNetwork", config=config)
        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        activation = (config or {}).get('activation', 'relu')
        dropout = (config or {}).get('dropout', 0.2)
        self.dropout_prob = dropout

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

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
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.train()
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() if self.output_size == 1 else nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            for i in range(0, len(x_tensor), batch_size):
                batch_x = x_tensor[i:i + batch_size]
                batch_y = y_tensor[i:i + batch_size]
                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            if epoch % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                logger.info(
                    "Epoch %d, Average Loss: %.4f",
                    epoch,
                    avg_loss
                )
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
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            outputs = self.forward(x_tensor)
            return outputs.numpy()

    def save(self, filepath: str) -> None:
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_prob': self.dropout_prob,
            'is_trained': self.is_trained
        }, filepath)
        logger.info("Model saved to %s", filepath)

    def load(self, filepath: str) -> None:
        """Load model from file"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        logger.info("Model loaded from %s", filepath)
