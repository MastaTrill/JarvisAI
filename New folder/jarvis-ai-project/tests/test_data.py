import unittest
from src.data.loaders import load_dataset
from src.data.preprocessing import normalize_data, split_data

class TestDataFunctions(unittest.TestCase):

    def test_load_dataset(self):
        # Test loading a dataset from a CSV file
        data = load_dataset('path/to/dataset.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_normalize_data(self):
        # Test normalization of data
        sample_data = [1, 2, 3, 4, 5]
        normalized = normalize_data(sample_data)
        self.assertEqual(len(normalized), len(sample_data))
        self.assertAlmostEqual(sum(normalized), 1.0)

    def test_split_data(self):
        # Test splitting data into training and testing sets
        data = [1, 2, 3, 4, 5]
        train, test = split_data(data, test_size=0.2)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 1)

if __name__ == '__main__':
    unittest.main()