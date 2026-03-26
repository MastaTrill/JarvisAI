import unittest
from src.data.loaders import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()

    def test_load_data(self):
        # Test loading data functionality
        data = self.data_loader.load_data('path/to/dataset')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_split_data(self):
        # Test data splitting functionality
        data = self.data_loader.load_data('path/to/dataset')
        train_data, val_data, test_data = self.data_loader.split_data(data)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertGreater(len(test_data), 0)

    def test_invalid_data_path(self):
        # Test loading data with an invalid path
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_data('invalid/path')

if __name__ == '__main__':
    unittest.main()