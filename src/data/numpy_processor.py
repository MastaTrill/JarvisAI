"""Numpy-based data processing utilities.

This module was referenced throughout training scripts and tests but was
missing from the tree after the src layout move. The API below follows
those existing callers.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DataProcessor:
    """Load, split, and scale tabular data using numpy/pandas only."""

    def __init__(
        self,
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
    ) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize
        self.is_fitted = False
        self.scaler_stats: Optional[Dict[str, np.ndarray]] = None

    def load_sample_data(self, n_samples: int = 100, n_features: int = 4) -> Dict[str, np.ndarray]:
        """Generate a small deterministic sample dataset."""
        rng = np.random.default_rng(self.random_state)
        data = rng.normal(size=(n_samples, n_features))
        weights = rng.normal(size=n_features)
        target = data @ weights + rng.normal(scale=0.1, size=n_samples)
        return {"data": data, "target": target}

    def create_dummy_data(self, n_samples: int = 100, n_features: int = 5) -> pd.DataFrame:
        """Return a DataFrame with feature_* columns and a target column."""
        rng = np.random.default_rng(self.random_state)
        features = rng.normal(size=(n_samples, n_features))
        weights = rng.normal(size=n_features)
        target = features @ weights + rng.normal(scale=0.1, size=n_samples)
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        frame[self.target_column] = target
        return frame

    def prepare_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split features and target into train and test sets."""
        X = np.asarray(data, dtype=float)
        if target is None:
            raise ValueError("target is required")
        y = np.asarray(target)
        if len(X) != len(y):
            raise ValueError("data and target must have the same length")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(X))
        test_count = max(1, int(round(len(X) * self.test_size)))
        test_count = min(test_count, len(X) - 1)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def scale_features(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Standardize features to mean 0 and standard deviation 1."""
        data = np.asarray(data, dtype=float)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1.0, std)
        stats = {"mean": mean, "std": std}
        self.scaler_stats = stats
        self.is_fitted = True
        return self.apply_scaling(data, stats), stats

    def apply_scaling(self, data: np.ndarray, scaler_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply previously computed mean/std scaling."""
        data = np.asarray(data, dtype=float)
        std = np.asarray(scaler_stats["std"], dtype=float)
        std = np.where(std == 0, 1.0, std)
        return (data - np.asarray(scaler_stats["mean"], dtype=float)) / std

    def process_pipeline(
        self, data_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a CSV, split, and scale features."""
        frame = pd.read_csv(data_path)
        if self.target_column not in frame.columns:
            raise ValueError(f"target column {self.target_column!r} not found")
        y = frame[self.target_column].to_numpy()
        X = frame.drop(columns=[self.target_column]).to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        if self.normalize:
            X_train, stats = self.scale_features(X_train)
            X_test = self.apply_scaling(X_test, stats)
        else:
            self.is_fitted = True
        return X_train, X_test, y_train, y_test

    def save_processor(self, payload: Any, filepath: str) -> None:
        """Serialize a processor payload to disk."""
        with open(filepath, "wb") as handle:
            pickle.dump(payload, handle)

    def load_processor(self, filepath: str) -> Any:
        """Load a processor payload from disk."""
        with open(filepath, "rb") as handle:
            return pickle.load(handle)

    def save_scaler(self, filepath: str) -> None:
        """Save the last computed scaler statistics."""
        if self.scaler_stats is None:
            raise ValueError("scaler has not been fitted")
        self.save_processor(self.scaler_stats, filepath)
