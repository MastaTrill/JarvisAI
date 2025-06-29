from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_data(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """Normalize numeric features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_features (list): List of numeric feature names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized numeric features.
    """
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df

def encode_categorical_data(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """Encode categorical features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical feature names to encode.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(categorical_features, axis=1)
    return pd.concat([df, encoded_df], axis=1)

def split_dataset(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The name of the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training features, testing features, training target, testing target.
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(df: pd.DataFrame, target: str, numeric_features: list, categorical_features: list) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data by normalizing and encoding features, and splitting the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The name of the target variable.
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target variable.
    """
    df = normalize_data(df, numeric_features)
    df = encode_categorical_data(df, categorical_features)
    return split_dataset(df, target)