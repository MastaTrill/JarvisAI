"""
Unit tests for the Jarvis AI Project inference module (numpy-only).
"""

import numpy as np
import tempfile
import os

from src.models.numpy_neural_network import SimpleNeuralNetwork
from src.data.numpy_processor import DataProcessor


class TestNumpyInference:
    """Test cases for numpy-only inference"""

    def setup_method(self):
        self.processor = DataProcessor(target_column="target")

    def test_model_creation(self):
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[10, 5], output_size=1)
        assert model.input_size == 4
        assert model.hidden_sizes == [10, 5]
        assert model.output_size == 1
        assert len(model.weights) == 3
        assert len(model.biases) == 3

    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[10, 5], output_size=1)

        # Test single sample
        X = np.random.randn(1, 4)
        output = model.forward(X)
        assert output.shape == (1, 1)

        # Test batch
        X_batch = np.random.randn(5, 4)
        output_batch = model.forward(X_batch)
        assert output_batch.shape == (5, 1)

    def test_model_predict(self):
        """Test model prediction"""
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[10, 5], output_size=1)

        # Test prediction
        X = np.random.randn(5, 4)
        predictions = model.predict(X)
        assert predictions.shape == (5, 1)
        assert isinstance(predictions, np.ndarray)

    def test_model_save_load(self):
        """Test model saving and loading"""
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[10, 5], output_size=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "test_model.pkl")
            model.save(model_path)
            assert os.path.exists(model_path)

            # Load model
            loaded_model = SimpleNeuralNetwork.load(model_path)
            assert loaded_model.input_size == model.input_size
            assert loaded_model.hidden_sizes == model.hidden_sizes
            assert loaded_model.output_size == model.output_size

            # Test that loaded model produces same output
            X = np.random.randn(3, 4)
            original_output = model.predict(X)
            loaded_output = loaded_model.predict(X)
            assert np.allclose(original_output, loaded_output)

    def test_inference_pipeline(self):
        """Test complete inference pipeline"""
        # Create and train a simple model
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[8, 4], output_size=1)

        # Get sample data
        data = self.processor.load_sample_data()
        X = data["data"][:10]  # Use first 10 samples

        # Make predictions
        predictions = model.predict(X)
        assert predictions.shape == (10, 1)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()

    def test_predict_with_model_function(self):
        """Test model prediction with saved/loaded model and preprocessor"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a model
            model = SimpleNeuralNetwork(
                input_size=4, hidden_sizes=[8, 4], output_size=1
            )
            model_path = os.path.join(temp_dir, "model.pkl")
            model.save(model_path)

            # Create and save preprocessor
            data = self.processor.load_sample_data()
            X_train = data["data"][:50]
            _, scaler_stats = self.processor.scale_features(X_train)

            preprocessor_path = os.path.join(temp_dir, "preprocessor.pkl")
            self.processor.save_processor(scaler_stats, preprocessor_path)

            # Load model and preprocessor, then predict
            loaded_model = SimpleNeuralNetwork.load(model_path)
            loaded_stats = self.processor.load_processor(preprocessor_path)
            X_test = data["data"][50:55]
            X_scaled = self.processor.apply_scaling(X_test, loaded_stats)
            predictions = loaded_model.predict(X_scaled)

            assert predictions.shape == (5, 1)
            assert isinstance(predictions, np.ndarray)
            assert not np.isnan(predictions).any()

    def test_batch_prediction(self):
        """Test batch prediction with different sizes"""
        model = SimpleNeuralNetwork(input_size=4, hidden_sizes=[8, 4], output_size=1)

        # Test different batch sizes
        for batch_size in [1, 5, 10, 50]:
            X = np.random.randn(batch_size, 4)
            predictions = model.predict(X)
            assert predictions.shape == (batch_size, 1)

    def test_model_with_different_architectures(self):
        """Test models with different architectures"""
        architectures = [
            (2, [5], 1),
            (3, [10, 5], 1),
            (4, [15, 10, 5], 1),
            (5, [20, 15, 10, 5], 1),
        ]

        for input_size, hidden_sizes, output_size in architectures:
            model = SimpleNeuralNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
            )

            # Test forward pass
            X = np.random.randn(3, input_size)
            output = model.forward(X)
            assert output.shape == (3, output_size)

            # Test prediction
            predictions = model.predict(X)
            assert predictions.shape == (3, output_size)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
