"""
Unit tests for the Jarvis AI Project training module (numpy-only).
"""

import pytest
import numpy as np
import tempfile
import os

from src.models.numpy_neural_network import NumpyNeuralNetwork
from src.training.numpy_trainer import NumpyTrainer
from src.data.numpy_processor import NumpyDataProcessor


class TestNumpyTraining:
    """Test cases for numpy-only training"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = NumpyDataProcessor(target_column='target')
        
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        
        trainer = NumpyTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=32
        )
        
        assert trainer.model == model
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 32
    
    def test_loss_calculation(self):
        """Test loss calculation"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        trainer = NumpyTrainer(model=model)
        
        # Test loss calculation
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [1.9], [3.1]])
        
        loss = trainer.compute_loss(y_true, y_pred)
        assert isinstance(loss, float)
        assert loss > 0
        
        # Test perfect prediction
        perfect_loss = trainer.compute_loss(y_true, y_true)
        assert perfect_loss < loss
    
    def test_batch_creation(self):
        """Test batch creation for training"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        trainer = NumpyTrainer(model=model, batch_size=5)
        
        # Create sample data
        X = np.random.randn(20, 4)
        y = np.random.randn(20, 1)
        
        # Get batches
        batches = trainer.create_batches(X, y)
        
        assert len(batches) == 4  # 20 samples / 5 batch_size = 4 batches
        
        for batch_X, batch_y in batches:
            assert batch_X.shape[0] == 5
            assert batch_y.shape[0] == 5
            assert batch_X.shape[1] == 4
            assert batch_y.shape[1] == 1
    
    def test_single_epoch_training(self):
        """Test single epoch training"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        trainer = NumpyTrainer(model=model, learning_rate=0.01, batch_size=5)
        
        # Get sample data
        data = self.processor.load_sample_data()
        X = data['data'][:20]
        y = data['target'][:20].reshape(-1, 1)
        
        # Train for one epoch
        initial_loss = trainer.compute_loss(y, model.forward(X))
        epoch_loss = trainer.train_epoch(X, y)
        final_loss = trainer.compute_loss(y, model.forward(X))
        
        assert isinstance(epoch_loss, float)
        assert epoch_loss > 0
        # Loss should generally decrease (though may not always due to randomness)
        assert isinstance(final_loss, float)
    
    def test_full_training(self):
        """Test full training process"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[6, 3],
            output_size=1
        )
        trainer = NumpyTrainer(
            model=model,
            learning_rate=0.01,
            batch_size=10
        )
        
        # Get sample data
        data = self.processor.load_sample_data()
        X_train = data['data'][:50]
        y_train = data['target'][:50].reshape(-1, 1)
        X_val = data['data'][50:70]
        y_val = data['target'][50:70].reshape(-1, 1)
        
        # Train model
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=5
        )
        
        # Check training history
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        
        # Check that losses are reasonable
        for loss in history['train_loss']:
            assert isinstance(loss, float)
            assert loss > 0
        for loss in history['val_loss']:
            assert isinstance(loss, float)
            assert loss > 0
    
    def test_model_improvement(self):
        """Test that model actually learns (loss decreases)"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[10, 5],
            output_size=1
        )
        trainer = NumpyTrainer(
            model=model,
            learning_rate=0.1,  # Higher learning rate for faster improvement
            batch_size=20
        )
        
        # Create linearly separable data for easier learning
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3]).reshape(-1, 1)
        
        # Initial loss
        initial_loss = trainer.compute_loss(y, model.forward(X))
        
        # Train for multiple epochs
        history = trainer.train(X, y, epochs=20)
        
        # Final loss
        final_loss = trainer.compute_loss(y, model.forward(X))
        
        # Loss should decrease significantly
        assert final_loss < initial_loss
        
        # Training loss should generally trend downward
        train_losses = history['train_loss']
        early_avg = np.mean(train_losses[:5])
        late_avg = np.mean(train_losses[-5:])
        assert late_avg < early_avg
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        model = NumpyNeuralNetwork(
            input_size=2,
            hidden_sizes=[3],
            output_size=1
        )
        trainer = NumpyTrainer(model=model)
        
        # Simple test case
        X = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])
        
        # Compute gradients
        gradients = trainer.compute_gradients(X, y)
        
        # Check that gradients exist for all parameters
        assert len(gradients['weights']) == len(model.weights)
        assert len(gradients['biases']) == len(model.biases)
        
        # Check gradient shapes match parameter shapes
        for i, (grad_w, w) in enumerate(zip(gradients['weights'], model.weights)):
            assert grad_w.shape == w.shape, f"Weight gradient {i} shape mismatch"
        
        for i, (grad_b, b) in enumerate(zip(gradients['biases'], model.biases)):
            assert grad_b.shape == b.shape, f"Bias gradient {i} shape mismatch"
    
    def test_parameter_update(self):
        """Test parameter update"""
        model = NumpyNeuralNetwork(
            input_size=2,
            hidden_sizes=[3],
            output_size=1
        )
        trainer = NumpyTrainer(model=model, learning_rate=0.1)
        
        # Store initial parameters
        initial_weights = [w.copy() for w in model.weights]
        initial_biases = [b.copy() for b in model.biases]
        
        # Simple training step
        X = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])
        
        # Compute gradients and update
        gradients = trainer.compute_gradients(X, y)
        trainer.update_parameters(gradients)
        
        # Check that parameters have changed
        for i, (initial_w, current_w) in enumerate(zip(initial_weights, model.weights)):
            assert not np.allclose(initial_w, current_w), f"Weight {i} did not update"
        
        for i, (initial_b, current_b) in enumerate(zip(initial_biases, model.biases)):
            assert not np.allclose(initial_b, current_b), f"Bias {i} did not update"
    
    def test_training_with_validation(self):
        """Test training with validation data"""
        model = NumpyNeuralNetwork(
            input_size=4,
            hidden_sizes=[8, 4],
            output_size=1
        )
        trainer = NumpyTrainer(model=model, learning_rate=0.01, batch_size=10)
        
        # Get sample data
        data = self.processor.load_sample_data()
        X_train = data['data'][:60]
        y_train = data['target'][:60].reshape(-1, 1)
        X_val = data['data'][60:80]
        y_val = data['target'][60:80].reshape(-1, 1)
        
        # Train with validation
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=3
        )
        
        # Check that both training and validation losses are recorded
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3


if __name__ == '__main__':
    pytest.main([__file__])
