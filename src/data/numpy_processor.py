"""
Data processing utilities using only numpy and pandas
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


class StandardScaler:
    """
    Simple standard scaler implementation using numpy
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit the scaler to the data"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        assert self.scale_ is not None
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    Simple min-max scaler implementation using numpy
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)) -> None:
        self.feature_range: Tuple[float, float] = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Fit the scaler to the data"""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        assert self.min_ is not None and self.max_ is not None
        # Avoid division by zero
        self.max_[self.max_ == self.min_] = self.min_[self.max_ == self.min_] + 1.0
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming")
        assert self.min_ is not None and self.max_ is not None
        X_std = (X - self.min_) / (self.max_ - self.min_)
        return (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        assert self.min_ is not None and self.max_ is not None
        X_std = (X - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        return X_std * (self.max_ - self.min_) + self.min_


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple train-test split implementation
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


class DataProcessor:
    """
    Data processor using only numpy and pandas
    """

    def __init__(
        self,
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the DataProcessor.

        Args:
            target_column (str): Name of the target column in the dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducible results.
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(
                "Successfully loaded data from %s. Shape: %s", file_path, data.shape
            )
            return data
        except Exception as e:
            logger.error("Error loading data from %s: %s", file_path, e)
            raise

    def preprocess_data(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the data by separating features and target, and scaling features.

        Args:
            data (pd.DataFrame): Raw input data.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Processed features and target arrays.
        """
        # Separate features and target
        if self.target_column in data.columns:
            X = data.drop(columns=[self.target_column])
            y = np.asarray(data[self.target_column].values)
        else:
            # If no target column, assume it's inference data
            X = data
            y = None

        # Convert to numpy and handle any non-numeric data
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.shape[1] != X.shape[1]:
            logger.warning("Non-numeric columns detected and removed")

        X_values = X_numeric.values

        # Scale features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_values)
            self.is_fitted = True
            logger.info("Fitted scaler to training data")
        else:
            X_scaled = self.scaler.transform(X_values)

        return X_scaled, y

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target array.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(
            "Data split completed. Train: %s, Test: %s", X_train.shape, X_test.shape
        )
        return X_train, X_test, y_train, y_test

    def process_pipeline(
        self, file_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data processing pipeline: load, preprocess, and split.

        Args:
            file_path (str): Path to the data file.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        # Load data
        data = self.load_data(file_path)

        # Preprocess data
        X, y = self.preprocess_data(data)

        if y is None:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        return X_train, X_test, y_train, y_test

    def save_scaler(self, file_path: str) -> None:
        """
        Save the fitted scaler to disk.

        Args:
            file_path (str): Path to save the scaler.
        """
        if not self.is_fitted:
            raise ValueError(
                "Scaler has not been fitted yet. Call preprocess_data first."
            )

        with open(file_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("Scaler saved to %s", file_path)

    def load_scaler(self, file_path: str) -> None:
        """
        Load a fitted scaler from disk.

        Args:
            file_path (str): Path to the saved scaler.
        """
        with open(file_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info("Scaler loaded from %s", file_path)

    def create_dummy_data(
        self, n_samples: int = 1000, n_features: int = 10
    ) -> pd.DataFrame:
        """
        Create dummy data for testing and demonstration purposes.

        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features to generate.

        Returns:
            pd.DataFrame: Generated dummy data.
        """
        np.random.seed(self.random_state)

        # Generate random features
        X = np.random.randn(n_samples, n_features)

        # Generate target as a simple linear combination with noise
        weights = np.random.randn(n_features)
        y = X @ weights + np.random.randn(n_samples) * 0.1

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data[self.target_column] = y

        logger.info("Generated dummy data with shape: %s", data.shape)
        return data

    def load_sample_data(self) -> dict:
        """
        Load sample data for testing.

        Returns:
            dict with 'data' (np.ndarray) and 'target' (np.ndarray) keys.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = 100, 4
        X = np.random.randn(n_samples, n_features)
        weights = np.array([1.0, -0.5, 0.3, 0.8])
        y = X @ weights + np.random.randn(n_samples) * 0.1
        return {"data": X, "target": y}

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets.

        Args:
            X: Feature array.
            y: Target array.

        Returns:
            Tuple of X_train, X_test, y_train, y_test.
        """
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def scale_features(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Scale features to zero mean and unit variance.

        Args:
            data: Feature array to scale.

        Returns:
            Tuple of (scaled_data, scaler_stats dict with 'mean' and 'std').
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1.0
        scaled = (data - mean) / std
        return scaled, {"mean": mean, "std": std}

    def apply_scaling(self, data: np.ndarray, scaler_stats: dict) -> np.ndarray:
        """
        Apply previously computed scaling to new data.

        Args:
            data: Feature array to scale.
            scaler_stats: Dict with 'mean' and 'std' from scale_features().

        Returns:
            Scaled data array.
        """
        return (data - scaler_stats["mean"]) / scaler_stats["std"]

    def save_processor(self, data: object, file_path: str) -> None:
        """
        Save arbitrary data (e.g. scaler stats) to a pickle file.

        Args:
            data: Object to save.
            file_path: Destination path.
        """
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Processor data saved to %s", file_path)

    def load_processor(self, file_path: str) -> object:
        """
        Load previously saved processor data.

        Args:
            file_path: Path to the pickle file.

        Returns:
            The loaded object.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Processor data loaded from %s", file_path)
        return data
