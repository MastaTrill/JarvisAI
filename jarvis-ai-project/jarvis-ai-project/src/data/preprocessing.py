from typing import Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)

def normalize_data(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize the input data using the specified method.

    Parameters:
    - data (pd.DataFrame): The input data to normalize.
    - method (str): The normalization method ('standard' or 'minmax').

    Returns:
    - pd.DataFrame: The normalized data.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logging.error("Invalid normalization method specified.")
        raise ValueError("Method must be 'standard' or 'minmax'.")

    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

def augment_data(data: np.ndarray, augmentations: Any) -> np.ndarray:
    """
    Apply data augmentations to the input data.

    Parameters:
    - data (np.ndarray): The input data to augment.
    - augmentations (Any): The augmentation transformations to apply.

    Returns:
    - np.ndarray: The augmented data.
    """
    transform = transforms.Compose(augmentations)
    augmented_data = transform(data)
    return augmented_data

def split_data(data: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Parameters:
    - data (pd.DataFrame): The input data to split.
    - train_size (float): The proportion of the data to include in the training set.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: The training and validation data.
    """
    train_data = data.sample(frac=train_size, random_state=42)
    val_data = data.drop(train_data.index)
    return train_data, val_data