"""
Comprehensive test suite for the train_simple.py training script.
Tests the complete training pipeline including configuration loading,
data processing, model training, and error handling.
"""

import pytest
import tempfile
import os
import yaml
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.train_simple import run_training
from src.data.numpy_processor import DataProcessor
from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.training.numpy_trainer import NumpyTrainer


class TestTrainSimple:
    """Test suite for the train_simple.py training script"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing"""
        return {
            "data": {
                "path": "data/sample_data.csv",
                "target_column": "target",
                "test_size": 0.2
            },
            "model": {
                "type": "SimpleNeuralNetwork",
                "hidden_sizes": [64, 32],
                "output_size": 1,
                "alpha": 0.0001
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.01,
                "seed": 42,
                "preprocessor_path": "artifacts/preprocessor.pkl",
                "model_path": "models/trained_model.pkl"
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target as a linear combination with some noise
        weights = np.random.randn(n_features)
        y = X @ weights + 0.1 * np.random.randn(n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        return data

    def test_run_training_success(self, sample_config, sample_data):
        """Test successful training pipeline execution"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Create data file
            data_path = os.path.join(temp_dir, 'sample_data.csv')
            sample_data.to_csv(data_path, index=False)
            sample_config['data']['path'] = data_path
            
            # Update config file with correct data path
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock the components to avoid actual training
            with patch('src.training.train_simple.DataProcessor') as mock_processor, \
                 patch('src.training.train_simple.SimpleNeuralNetwork') as mock_model, \
                 patch('src.training.train_simple.NumpyTrainer') as mock_trainer:
                
                # Configure mocks
                mock_processor_instance = MagicMock()
                mock_processor.return_value = mock_processor_instance
                mock_processor_instance.process_pipeline.return_value = (
                    np.random.randn(80, 5),  # X_train
                    np.random.randn(20, 5),  # X_test
                    np.random.randn(80),     # y_train
                    np.random.randn(20)      # y_test
                )
                
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance
                mock_trainer_instance.train.return_value = {
                    'train_score': 0.85,
                    'test_score': 0.82,
                    'training_time': 1.23
                }
                
                # Run training - should not raise any exceptions
                run_training(config_path)
                
                # Verify components were called
                mock_processor.assert_called_once()
                mock_model.assert_called_once()
                mock_trainer.assert_called_once()
                mock_trainer_instance.train.assert_called_once()

    def test_run_training_config_file_not_found(self):
        """Test behavior when configuration file doesn't exist"""
        non_existent_config = "non_existent_config.yaml"
        
        # Should not raise an exception, but should log error
        with patch('src.training.train_simple.logger') as mock_logger:
            run_training(non_existent_config)
            mock_logger.error.assert_called_once()

    def test_run_training_invalid_yaml(self):
        """Test behavior when configuration file has invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with patch('src.training.train_simple.logger') as mock_logger:
                run_training(config_path)
                mock_logger.error.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_run_training_missing_data_file(self, sample_config):
        """Test behavior when data file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with non-existent data path
            sample_config['data']['path'] = "non_existent_data.csv"
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock DataProcessor to raise FileNotFoundError
            with patch('src.training.train_simple.DataProcessor') as mock_processor:
                mock_processor_instance = MagicMock()
                mock_processor.return_value = mock_processor_instance
                mock_processor_instance.process_pipeline.side_effect = FileNotFoundError("Data file not found")
                
                with patch('src.training.train_simple.logger') as mock_logger:
                    run_training(config_path)
                    # Should log error about data loading
                    mock_logger.error.assert_called()

    def test_run_training_model_training_failure(self, sample_config, sample_data):
        """Test behavior when model training fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config and data files
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            data_path = os.path.join(temp_dir, 'sample_data.csv')
            sample_data.to_csv(data_path, index=False)
            sample_config['data']['path'] = data_path
            
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock components with training failure
            with patch('src.training.train_simple.DataProcessor') as mock_processor, \
                 patch('src.training.train_simple.SimpleNeuralNetwork') as mock_model, \
                 patch('src.training.train_simple.NumpyTrainer') as mock_trainer:
                
                # Configure successful data processing
                mock_processor_instance = MagicMock()
                mock_processor.return_value = mock_processor_instance
                mock_processor_instance.process_pipeline.return_value = (
                    np.random.randn(80, 5),
                    np.random.randn(20, 5),
                    np.random.randn(80),
                    np.random.randn(20)
                )
                
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                
                # Configure training failure
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance
                mock_trainer_instance.train.side_effect = RuntimeError("Training failed")
                
                with patch('src.training.train_simple.logger') as mock_logger:
                    run_training(config_path)
                    # Should log error about training failure
                    mock_logger.error.assert_called()

    def test_run_training_configuration_validation(self, sample_config):
        """Test validation of configuration parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            
            # Test missing required sections
            incomplete_configs = [
                {},  # Empty config
                {"data": {}},  # Missing model and training
                {"data": {"path": "test.csv"}, "model": {}},  # Missing training
            ]
            
            for incomplete_config in incomplete_configs:
                with open(config_path, 'w') as f:
                    yaml.dump(incomplete_config, f)
                
                # Should handle missing configuration gracefully
                with patch('src.training.train_simple.logger') as mock_logger:
                    run_training(config_path)
                    # May log warnings or errors about missing config

    def test_run_training_with_different_model_types(self, sample_config, sample_data):
        """Test training with different model configurations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, 'sample_data.csv')
            sample_data.to_csv(data_path, index=False)
            sample_config['data']['path'] = data_path
            
            # Test different model configurations
            model_configs = [
                {"type": "SimpleNeuralNetwork", "hidden_sizes": [32], "output_size": 1, "alpha": 0.0001},
                {"type": "SimpleNeuralNetwork", "hidden_sizes": [64, 32], "output_size": 1, "alpha": 0.001},
            ]
            
            for model_config in model_configs:
                sample_config['model'] = model_config
                config_path = os.path.join(temp_dir, f'test_config_{len(model_config["hidden_sizes"])}.yaml')
                
                with open(config_path, 'w') as f:
                    yaml.dump(sample_config, f)
                
                # Mock successful training
                with patch('src.training.train_simple.DataProcessor') as mock_processor, \
                     patch('src.training.train_simple.SimpleNeuralNetwork') as mock_model, \
                     patch('src.training.train_simple.NumpyTrainer') as mock_trainer:
                    
                    mock_processor_instance = MagicMock()
                    mock_processor.return_value = mock_processor_instance
                    mock_processor_instance.process_pipeline.return_value = (
                        np.random.randn(80, 5),
                        np.random.randn(20, 5),
                        np.random.randn(80),
                        np.random.randn(20)
                    )
                    
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    
                    mock_trainer_instance = MagicMock()
                    mock_trainer.return_value = mock_trainer_instance
                    mock_trainer_instance.train.return_value = {
                        'train_score': 0.85,
                        'test_score': 0.82
                    }
                    
                    # Should complete successfully
                    run_training(config_path)
                    mock_trainer_instance.train.assert_called_once()

    def test_run_training_logging(self, sample_config, sample_data):
        """Test that appropriate logging occurs during training"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            data_path = os.path.join(temp_dir, 'sample_data.csv')
            sample_data.to_csv(data_path, index=False)
            sample_config['data']['path'] = data_path
            
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            with patch('src.training.train_simple.DataProcessor') as mock_processor, \
                 patch('src.training.train_simple.SimpleNeuralNetwork') as mock_model, \
                 patch('src.training.train_simple.NumpyTrainer') as mock_trainer, \
                 patch('src.training.train_simple.logger') as mock_logger:
                
                # Configure successful mocks
                mock_processor_instance = MagicMock()
                mock_processor.return_value = mock_processor_instance
                mock_processor_instance.process_pipeline.return_value = (
                    np.random.randn(80, 5),
                    np.random.randn(20, 5),
                    np.random.randn(80),
                    np.random.randn(20)
                )
                
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                
                mock_trainer_instance = MagicMock()
                mock_trainer.return_value = mock_trainer_instance
                mock_trainer_instance.train.return_value = {
                    'train_score': 0.85,
                    'test_score': 0.82
                }
                
                run_training(config_path)
                
                # Verify logging calls
                mock_logger.info.assert_called()  # Should log successful operations
