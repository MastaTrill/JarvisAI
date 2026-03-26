import unittest
from src.data.preprocessing import normalize, augment_data
from src.data.loaders import load_data

class TestDataProcessing(unittest.TestCase):

    def test_normalize(self):
        data = [1, 2, 3, 4, 5]
        normalized_data = normalize(data)
        self.assertAlmostEqual(normalized_data[0], 0.0)
        self.assertAlmostEqual(normalized_data[-1], 1.0)

    def test_augment_data(self):
        original_data = [1, 2, 3]
        augmented_data = augment_data(original_data)
        self.assertGreater(len(augmented_data), len(original_data))

    def test_load_data(self):
        data = load_data('path/to/dataset.csv')
        self.assertIsNotNone(data)
        self.assertTrue(len(data) > 0)

if __name__ == '__main__':
    unittest.main()