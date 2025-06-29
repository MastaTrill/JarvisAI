import unittest
import pandas as pd
import numpy as np
import sys
import os
import importlib.util

# Dynamically import preprocessors.py for robust path handling
preprocessors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/data/preprocessors.py'))
spec = importlib.util.spec_from_file_location("preprocessors", preprocessors_path)
preprocessors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessors)
normalize_data = preprocessors.normalize_data
split_data = preprocessors.split_data

class TestPreprocessors(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })

    def test_normalize_data(self):
        norm = normalize_data(self.df)
        self.assertTrue(isinstance(norm, pd.DataFrame))
        # Mean should be close to 0, std close to 1
        self.assertTrue(np.allclose(norm.mean(), 0, atol=1e-8))
        self.assertTrue(np.allclose(norm.std(), 1, atol=1e-8))

    def test_normalize_data_zero_std(self):
        df = pd.DataFrame({'a': [1, 1, 1]})
        norm = normalize_data(df)
        self.assertTrue((norm == 0).all().all())

    def test_split_data(self):
        train, val = split_data(self.df, train_size=0.6)
        self.assertEqual(len(train) + len(val), len(self.df))
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 2)
        self.assertTrue(set(train.index).isdisjoint(val.index))

    def test_split_data_invalid_type(self):
        with self.assertRaises(TypeError):
            split_data([1, 2, 3], train_size=0.5)

    def test_split_data_invalid_size(self):
        with self.assertRaises(ValueError):
            split_data(self.df, train_size=1.5)

if __name__ == '__main__':
    unittest.main()
