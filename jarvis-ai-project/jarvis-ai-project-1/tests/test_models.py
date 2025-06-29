import unittest
from src.models.base_model import BaseModel
from src.models.neural_networks import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.model = NeuralNetwork()

    def test_model_initialization(self):
        self.assertIsInstance(self.model, NeuralNetwork)

    def test_train_method(self):
        # Assuming the train method returns a boolean indicating success
        result = self.model.train()
        self.assertTrue(result)

    def test_evaluate_method(self):
        # Assuming the evaluate method returns a dictionary of metrics
        metrics = self.model.evaluate()
        self.assertIn('accuracy', metrics)
        self.assertIn('loss', metrics)

if __name__ == '__main__':
    unittest.main()