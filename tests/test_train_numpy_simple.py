"""
Test suite for the numpy-based training script (train_numpy_simple.py).
"""

import pytest
import tempfile
import os
import yaml
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.train_numpy_simple import run_numpy_training


class TestNumpyTraining:
    """Test suite for the numpy-based training script"""

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
                "hidden_sizes": [64, 32],
                "output_size": 1
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.01,
                "seed": 42,
                "model_path": "models/test_model.pkl"
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Generate random features and target
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y
        
        return data

    def test_run_numpy_training_success(self, sample_config, sample_data):
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
            
            # Update config with correct paths
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock the components
            with patch('src.training.train_numpy_simple.DataProcessor') as mock_processor, \
                 patch('src.training.train_numpy_simple.SimpleNeuralNetwork') as mock_model, \
                 patch('src.training.train_numpy_simple.NumpyTrainer') as mock_trainer:
                
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
                    'train_loss': 0.15,
                    'val_loss': 0.18,
                    'train_accuracy': 0.85,
                    'val_accuracy': 0.82
                }
                
                # Run training - should not raise any exceptions
                run_numpy_training(config_path)
                
                # Verify components were called
                mock_processor.assert_called_once()
                mock_model.assert_called_once()
                mock_trainer.assert_called_once()
                mock_trainer_instance.train.assert_called_once()

    def test_run_numpy_training_config_not_found(self):
        """Test behavior when configuration file doesn't exist"""
        non_existent_config = "non_existent_config.yaml"
        
        with patch('src.training.train_numpy_simple.logger') as mock_logger:
            run_numpy_training(non_existent_config)
            mock_logger.error.assert_called_once()

    def test_run_numpy_training_invalid_yaml(self):
        """Test behavior when configuration file has invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with patch('src.training.train_numpy_simple.logger') as mock_logger:
                run_numpy_training(config_path)
                mock_logger.error.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_run_numpy_training_data_file_not_found(self, sample_config):
        """Test behavior when data file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with non-existent data path
            sample_config['data']['path'] = "non_existent_data.csv"
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            with patch('src.training.train_numpy_simple.logger') as mock_logger:
                run_numpy_training(config_path)
                mock_logger.error.assert_called()

    def test_run_numpy_training_with_exception(self, sample_config, sample_data):
        """Test behavior when an exception occurs during training"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid config and data files
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            data_path = os.path.join(temp_dir, 'sample_data.csv')
            sample_data.to_csv(data_path, index=False)
            sample_config['data']['path'] = data_path
            
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Mock DataProcessor to raise an exception
            with patch('src.training.train_numpy_simple.DataProcessor') as mock_processor:
                mock_processor.side_effect = RuntimeError("Processing failed")
                
                with patch('src.training.train_numpy_simple.logger') as mock_logger:
                    run_numpy_training(config_path)
                    mock_logger.error.assert_called()

    def test_config_loading_and_validation(self, sample_config):
        """Test configuration loading and basic validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            
            # Test with valid config
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)
                
            with patch('src.training.train_numpy_simple.DataProcessor') as mock_processor:
                mock_processor.side_effect = FileNotFoundError("Data not found")
                
                with patch('src.training.train_numpy_simple.logger') as mock_logger:
                    run_numpy_training(config_path)
                    # Should log successful config loading
                    info_calls = [call for call in mock_logger.info.call_args_list 
                                if 'Configuration loaded successfully' in str(call)]
                    assert len(info_calls) > 0
