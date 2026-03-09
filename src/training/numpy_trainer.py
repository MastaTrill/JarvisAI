"""
Simple trainer for numpy-based neural networks
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle

logger = logging.getLogger(__name__)


class NumpyTrainer:
    """
    Simple trainer for numpy-based neural networks
    """

    def __init__(self, model, learning_rate: float = 0.001, batch_size: int = 32):
        """
        Initialize the trainer

        Args:
            model: The neural network model to train
            learning_rate: Learning rate for gradient descent
            batch_size: Mini-batch size
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_scores: List[float] = []
        self.val_scores: List[float] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        logger.info("Trainer initialized")

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error loss."""
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        return float(np.mean((y_true - y_pred) ** 2))

    def create_batches(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data into mini-batches."""
        n = len(X)
        batches = []
        for i in range(0, n, self.batch_size):
            batches.append((X[i : i + self.batch_size], y[i : i + self.batch_size]))
        return batches

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Dict[str, list]:
        """Compute gradients via backpropagation without updating weights."""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        activations, z_values = self.model._forward_pass(X)

        weight_grads: list = [None] * len(self.model.weights)
        bias_grads: list = [None] * len(self.model.biases)

        dA = activations[-1] - y

        for i in reversed(range(len(self.model.weights))):
            if i == len(self.model.weights) - 1:
                dZ = dA
            else:
                dZ = dA * self.model._relu_derivative(z_values[i])

            weight_grads[i] = (1 / m) * np.dot(activations[i].T, dZ)
            bias_grads[i] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            if i > 0:
                dA = np.dot(dZ, self.model.weights[i].T)

        return {"weights": weight_grads, "biases": bias_grads}

    def update_parameters(self, gradients: Dict[str, list]) -> None:
        """Apply gradient updates to model parameters."""
        for i in range(len(self.model.weights)):
            self.model.weights[i] -= self.learning_rate * gradients["weights"][i]
            self.model.biases[i] -= self.learning_rate * gradients["biases"][i]

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train for one epoch and return average loss."""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in self.create_batches(X_shuffled, y_shuffled):
            grads = self.compute_gradients(batch_X, batch_y)
            self.update_parameters(grads)

            pred = self.model.forward(batch_X)
            total_loss += self.compute_loss(batch_y, pred)
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
    ) -> Dict[str, list]:
        """
        Train the model, returning per-epoch loss history.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs

        Returns:
            Dict with 'train_loss' and 'val_loss' lists (one entry per epoch).
        """
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)

        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        logger.info("Starting training for %d epochs", epochs)

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(X_train, y_train)
            history["train_loss"].append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.model.forward(X_val)
                v_loss = self.compute_loss(y_val, val_pred)
            else:
                v_loss = epoch_loss
            history["val_loss"].append(v_loss)

        self.model.is_trained = True

        logger.info(
            "Training completed - final train_loss: %.6f", history["train_loss"][-1]
        )
        return history

    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        from src.models.numpy_neural_network import SimpleNeuralNetwork

        self.model = SimpleNeuralNetwork.load(filepath)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def get_training_history(self) -> Dict[str, list]:
        """Get training history"""
        return {
            "train_scores": self.train_scores,
            "val_scores": self.val_scores,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


# Alias for compatibility
Trainer = NumpyTrainer
