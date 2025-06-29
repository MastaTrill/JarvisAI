from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

def normalize_data(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize the input data using the specified method.

    Parameters:
    - data: pd.DataFrame - The input data to normalize.
    - method: str - The normalization method ('standard' or 'minmax').

    Returns:
    - pd.DataFrame - The normalized data.
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

def augment_data(data: pd.DataFrame, augment_factor: int) -> pd.DataFrame:
    """
    Augment the input data by duplicating it a specified number of times.

    Parameters:
    - data: pd.DataFrame - The input data to augment.
    - augment_factor: int - The number of times to duplicate the data.

    Returns:
    - pd.DataFrame - The augmented data.
    """
    if augment_factor < 1:
        logging.error("Augment factor must be at least 1.")
        raise ValueError("Augment factor must be at least 1.")

    augmented_data = pd.concat([data] * augment_factor, ignore_index=True)
    return augmented_data

def split_data(data: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Split the data into training, validation, and test sets.

    Parameters:
    - data: pd.DataFrame - The input data to split.
    - target: str - The name of the target variable.
    - test_size: float - The proportion of the dataset to include in the test split.
    - random_state: int - Random seed for reproducibility.

    Returns:
    - Dict[str, Any] - A dictionary containing the training, validation, and test sets.
    """
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }