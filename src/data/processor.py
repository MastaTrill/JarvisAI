"""
Data processing utilities for the Jarvis AI Project.

This module contains classes and functions for data preprocessing,
feature engineering, and data loading.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A comprehensive data processor for machine learning workflows.
    
    This class handles data loading, preprocessing, feature engineering,
    and train/validation/test splitting.
    """
    
    def __init__(self, target_column: str = 'target', test_size: float = 0.2, random_state: int = 42):
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
            logger.info(f"Successfully loaded data from {file_path}. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data by separating features and target, and scaling features.
        
        Args:
            data (pd.DataFrame): Raw input data.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed features and target arrays.
        """
        # Separate features and target
        if self.target_column in data.columns:
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column].values
        else:
            # If no target column, assume it's inference data
            X = data
            y = None
            
        # Scale features
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            logger.info("Fitted scaler to training data")
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            raise ValueError("Scaler has not been fitted yet. Call preprocess_data first.")
            
        joblib.dump(self.scaler, file_path)
        logger.info(f"Scaler saved to {file_path}")
    
    def load_scaler(self, file_path: str) -> None:
        """
        Load a fitted scaler from disk.
        
        Args:
            file_path (str): Path to the saved scaler.
        """
        self.scaler = joblib.load(file_path)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {file_path}")
    
    def create_dummy_data(self, n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
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
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(X, columns=feature_names)
        data[self.target_column] = y
        
        logger.info(f"Generated dummy data with shape: {data.shape}")
        return data
