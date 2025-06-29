"""
Unit tests for the Jarvis AI Project training module (numpy version).
"""

import pytest
import numpy as np
import tempfile
import os

from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.training.numpy_trainer import NumpyTrainer
from src.data.numpy_processor import DataProcessor


class TestSimpleNeuralNetwork:
    """Test cases for SimpleNeuralNetwork"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = SimpleNeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=1
        )
        assert model.input_size == 10
        assert model.hidden_sizes == [64, 32]
        assert model.output_size == 1
        assert not model.is_trained
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = SimpleNeuralNetwork(
            input_size=5,
            hidden_sizes=[10],
            output_size=1
        )
        x = np.random.randn(3, 5)
        activations, _ = model._forward_pass(x)
        assert activations[-1].shape == (3, 1)
    
    def test_model_fit_predict(self):
        """Test model training and prediction"""
        model = SimpleNeuralNetwork(
            input_size=5,
            hidden_sizes=[10],
            output_size=1
        )
        
        # Create dummy data
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        # Train model
        model.fit(X, y, epochs=5, batch_size=10)
        assert model.is_trained
        
        # Make predictions
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 1)
    
    def test_model_save_load(self):
        """Test model save and load"""
        model = SimpleNeuralNetwork(
            input_size=3,
            hidden_sizes=[5],
            output_size=1
        )
        
        # Train a bit
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model.fit(X, y, epochs=5)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            try:
                # Save model
                model.save(f.name)
                
                # Create new model and load
                new_model = SimpleNeuralNetwork(
                    input_size=3,
                    hidden_sizes=[5],
                    output_size=1
                )
                new_model.load(f.name)
                
                assert new_model.is_trained
                assert new_model.input_size == 3
                assert new_model.hidden_sizes == [5]
                
            finally:
                try:
                    os.unlink(f.name)
                except (PermissionError, FileNotFoundError):
                    # Windows file permission issue - not critical for test functionality
                    pass


class TestNumpyTrainer:
    """Test cases for NumpyTrainer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.model = SimpleNeuralNetwork(
            input_size=5,
            hidden_sizes=[10],
            output_size=1
        )
        self.trainer = NumpyTrainer(model=self.model)
        
        # Create dummy data
        self.X_train = np.random.randn(40, 5)
        self.y_train = np.random.randn(40)
        self.X_val = np.random.randn(10, 5)
        self.y_val = np.random.randn(10)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        assert self.trainer.model is not None
        assert len(self.trainer.train_scores) == 0
        assert len(self.trainer.val_scores) == 0
    
    def test_full_training(self):
        """Test full training loop"""
        metrics = self.trainer.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=5
        )
        
        assert 'final_train_score' in metrics
        assert 'final_val_score' in metrics
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        
        assert len(self.trainer.train_scores) == 1
        assert len(self.trainer.val_scores) == 1
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train first
        self.trainer.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=3
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            try:
                # Save model
                self.trainer.save_model(f.name)
                assert os.path.exists(f.name)
                
                # Load model
                new_trainer = NumpyTrainer(
                    model=SimpleNeuralNetwork(
                        input_size=5,
                        hidden_sizes=[10],
                        output_size=1
                    )
                )
                new_trainer.load_model(f.name)
                
                # Test prediction
                predictions = new_trainer.predict(self.X_val)
                assert predictions.shape == (10, 1)
                
            finally:
                try:
                    os.unlink(f.name)
                except (PermissionError, FileNotFoundError):
                    # Windows file permission issue - not critical for test functionality
                    pass


class TestDataProcessor:
    """Test cases for DataProcessor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = DataProcessor(target_column='target')
    
    def test_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor(target_column='y', test_size=0.3, random_state=123)
        assert processor.target_column == 'y'
        assert processor.test_size == 0.3
        assert processor.random_state == 123
        assert not processor.is_fitted
    
    def test_create_dummy_data(self):
        """Test dummy data generation"""
        data = self.processor.create_dummy_data(n_samples=100, n_features=5)
        
        # Check data structure
        assert data.shape == (100, 6)  # 5 features + 1 target
        assert 'target' in data.columns
        
        # Check feature columns
        feature_cols = [col for col in data.columns if col.startswith('feature_')]
        assert len(feature_cols) == 5
    
    def test_process_pipeline(self):
        """Test complete processing pipeline"""
        # Create temporary CSV file
        test_data = self.processor.create_dummy_data(n_samples=50, n_features=3)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            
            try:
                X_train, X_test, y_train, y_test = self.processor.process_pipeline(f.name)
                
                # Check shapes
                assert X_train.shape[1] == X_test.shape[1] == 3
                assert X_train.shape[0] + X_test.shape[0] == 50
                assert y_train.shape[0] + y_test.shape[0] == 50
                
                # Check processor is fitted
                assert self.processor.is_fitted
                
            finally:
                try:
                    os.unlink(f.name)
                except (PermissionError, FileNotFoundError):
                    # Windows file permission issue - not critical for test functionality
                    pass


if __name__ == "__main__":
    pytest.main([__file__])
