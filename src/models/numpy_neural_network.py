"""
Simple neural network implementation using only numpy
"""

import logging
import numpy as np
import pickle
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleNeuralNetwork:
    """
    Simple feedforward neural network using only numpy
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
            config: Optional dictionary with configuration
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.config = config or {}
        self.is_trained = False
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def _forward_pass(self, x):
        """Forward pass through the network"""
        activations = [x]
        z_values = []
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            
            if i < len(self.weights) - 1:  # Hidden layers use ReLU
                a = self._relu(z)
            else:  # Output layer is linear for regression
                a = z
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_pass(self, x, y, activations, z_values, learning_rate):
        """Backward pass (backpropagation)"""
        m = x.shape[0]
        
        # Calculate output layer error
        dA = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:  # Output layer
                dZ = dA
            else:  # Hidden layers
                dZ = dA * self._relu_derivative(z_values[i])
            
            # Calculate gradients
            dW = (1/m) * np.dot(activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Calculate error for next layer
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)

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
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        logger.info(f"Training neural network for {epochs} epochs")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(x_shuffled), batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Forward pass
                activations, z_values = self._forward_pass(batch_x)
                
                # Calculate loss (MSE)
                loss = np.mean((activations[-1] - batch_y) ** 2)
                total_loss += loss
                num_batches += 1
                
                # Backward pass
                self._backward_pass(batch_x, batch_y, activations, z_values, learning_rate)
            
            # Log progress
            if epoch % 20 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
        
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
        
        activations, _ = self._forward_pass(x)
        return activations[-1]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score for regression

        Args:
            x: Test features
            y: Test targets

        Returns:
            R² score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        predictions = self.predict(x)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Calculate R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return float(r2)

    def save(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.input_size = model_data['input_size']
        self.hidden_sizes = model_data['hidden_sizes']
        self.output_size = model_data['output_size']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
