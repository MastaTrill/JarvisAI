import unittest
from src.data.loaders import load_csv, load_image
from src.data.preprocessors import normalize_data, split_data
import pandas as pd
import numpy as np

class TestDataFunctions(unittest.TestCase):

    def test_load_csv(self):
        # Test loading a CSV file
        df = load_csv('path/to/test.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_load_image(self):
        # Test loading an image file
        image = load_image('path/to/test_image.jpg')
        self.assertIsNotNone(image)

    def test_normalize_data(self):
        # Test normalization of data
        data = np.array([[1, 2], [3, 4]])
        normalized = normalize_data(data)
        self.assertEqual(normalized.shape, data.shape)
        self.assertTrue(np.all(normalized >= 0) and np.all(normalized <= 1))

    def test_split_data(self):
        # Test splitting data into train and test sets
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        train, test = split_data(data, test_size=0.25)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 1)

if __name__ == '__main__':
    unittest.main()