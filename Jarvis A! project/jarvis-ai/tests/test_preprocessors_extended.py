import unittest
import pandas as pd
import numpy as np
from src.data.preprocessors import (
    impute_missing, normalize_data, split_data, encode_categorical, remove_outliers, preprocess_pipeline
)

class TestPreprocessorsExtended(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'cat': ['x', 'y', 'x', 'z', 'y']
        })

    def test_impute_missing_mean(self):
        df_imputed = impute_missing(self.df, strategy='mean')
        self.assertFalse(df_imputed.isnull().any().any())
        self.assertAlmostEqual(df_imputed['a'].iloc[2], self.df['a'].mean(skipna=True))

    def test_impute_missing_median(self):
        df_imputed = impute_missing(self.df, strategy='median')
        self.assertFalse(df_imputed.isnull().any().any())
        self.assertAlmostEqual(df_imputed['a'].iloc[2], self.df['a'].median(skipna=True))

    def test_impute_missing_mode(self):
        df_imputed = impute_missing(self.df, strategy='mode')
        self.assertFalse(df_imputed.isnull().any().any())
        self.assertEqual(df_imputed['cat'].iloc[2], 'x')

    def test_encode_categorical(self):
        df_encoded = encode_categorical(self.df, columns=['cat'])
        self.assertTrue('cat_x' in df_encoded.columns)
        self.assertTrue('cat_y' in df_encoded.columns)
        self.assertTrue('cat_z' in df_encoded.columns)

    def test_remove_outliers(self):
        df_no_outliers = remove_outliers(self.df[['a', 'b']].fillna(0), z_thresh=2.0)
        self.assertTrue(isinstance(df_no_outliers, pd.DataFrame))
        self.assertLessEqual(len(df_no_outliers), len(self.df))

    def test_preprocess_pipeline(self):
        df_processed = preprocess_pipeline(self.df)
        self.assertTrue(isinstance(df_processed, pd.DataFrame))
        self.assertFalse(df_processed.isnull().any().any())
        # Should be all numeric or boolean after encoding
        self.assertTrue(all(np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_) for dt in df_processed.dtypes))

if __name__ == '__main__':
    unittest.main()
