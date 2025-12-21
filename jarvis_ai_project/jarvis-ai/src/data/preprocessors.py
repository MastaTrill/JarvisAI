from typing import Tuple
import numpy as np
import pandas as pd

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the input DataFrame.

    Parameters:
    data (pd.DataFrame): The input data to normalize.

    Returns:
    pd.DataFrame: The normalized data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    std = data.std()
    # Avoid division by zero
    std_replaced = std.replace(0, 1)
    normalized = (data - data.mean()) / std_replaced
    return normalized

def split_data(data: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Parameters:
    data (pd.DataFrame): The input data to split.
    train_size (float): The proportion of the data to include in the training set.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and validation data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1.")
    train_data = data.sample(frac=train_size, random_state=42)
    val_data = data.drop(train_data.index)
    return train_data, val_data

def impute_missing(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input data with possible missing values.
    strategy (str): Imputation strategy ('mean', 'median', 'mode').

    Returns:
    pd.DataFrame: DataFrame with imputed values.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    df = data.copy()
    if strategy == 'mean':
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        return df
    elif strategy == 'median':
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        return df
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Invalid imputation strategy. Choose 'mean', 'median', or 'mode'.")

def encode_categorical(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    One-hot encode categorical columns in the DataFrame.

    Parameters:
    data (pd.DataFrame): The input data.
    columns (list or None): Columns to encode. If None, all object columns are encoded.

    Returns:
    pd.DataFrame: DataFrame with categorical columns encoded.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    return pd.get_dummies(data, columns=columns)

def remove_outliers(data: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Remove rows with outliers based on z-score.

    Parameters:
    data (pd.DataFrame): The input data.
    z_thresh (float): Z-score threshold for outlier removal.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    num_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((data[num_cols] - data[num_cols].mean()) / data[num_cols].std(ddof=0))
    mask = (z_scores < z_thresh).all(axis=1)
    return data[mask]

def preprocess_pipeline(data: pd.DataFrame, impute_strategy='mean', normalize=True, encode=True, outlier_removal=True) -> pd.DataFrame:
    """
    Full preprocessing pipeline: impute, remove outliers, encode, normalize.

    Parameters:
    data (pd.DataFrame): The input data.
    impute_strategy (str): Strategy for missing value imputation.
    normalize (bool): Whether to normalize numeric columns.
    encode (bool): Whether to one-hot encode categorical columns.
    outlier_removal (bool): Whether to remove outliers.

    Returns:
    pd.DataFrame: Preprocessed data.
    """
    df = impute_missing(data, strategy=impute_strategy)
    if outlier_removal:
        df = remove_outliers(df)
    if encode:
        df = encode_categorical(df)
    if normalize:
        # Only normalize numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = normalize_data(df[num_cols])
    return df