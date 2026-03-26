import unittest
from src.models.base_model import BaseModel
from src.models.neural_networks import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        self.model = NeuralNetwork()

    def test_initialization(self):
        self.assertIsInstance(self.model, NeuralNetwork)

    def test_train(self):
        # Assuming the train method returns a loss value
        loss = self.model.train()
        self.assertIsInstance(loss, float)

    def test_evaluate(self):
        # Assuming the evaluate method returns accuracy
        accuracy = self.model.evaluate()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()