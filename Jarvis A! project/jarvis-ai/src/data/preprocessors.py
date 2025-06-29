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
    return (data - data.mean()) / data.std()

def split_data(data: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Parameters:
    data (pd.DataFrame): The input data to split.
    train_size (float): The proportion of the data to include in the training set.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and validation data.
    """
    train_data = data.sample(frac=train_size, random_state=42)
    val_data = data.drop(train_data.index)
    return train_data, val_data