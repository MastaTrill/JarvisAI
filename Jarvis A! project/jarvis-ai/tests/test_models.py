import unittest
from src.models.base_model import BaseModel

class TestBaseModel(unittest.TestCase):

    def setUp(self):
        self.model = BaseModel()

    def test_training(self):
        # Assuming the BaseModel has a train method
        result = self.model.train()
        self.assertTrue(result)

    def test_evaluation(self):
        # Assuming the BaseModel has an evaluate method
        result = self.model.evaluate()
        self.assertIsInstance(result, dict)  # Assuming evaluation returns a dictionary of metrics

    def test_save_load(self):
        # Test saving and loading the model
        self.model.save('test_model.pth')
        loaded_model = BaseModel.load('test_model.pth')
        self.assertEqual(self.model.state_dict(), loaded_model.state_dict())

if __name__ == '__main__':
    unittest.main()