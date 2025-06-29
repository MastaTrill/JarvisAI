from typing import Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class DataLoader:
    def __init__(self, file_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the DataLoader with the specified file path and parameters.

        Parameters:
        - file_path (str): Path to the dataset file (CSV format).
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Random seed for reproducibility.
        """
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        logging.basicConfig(level=logging.INFO)

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified file path.

        Returns:
        - pd.DataFrame: Loaded dataset.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}.")
            return self.data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the loaded dataset into training, validation, and test sets.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test sets.
        """
        if self.data is None:
            logging.error("Data not loaded. Please load the data before splitting.")
            raise ValueError("Data not loaded.")

        train_data, test_data = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)
        logging.info("Data split into training and test sets.")
        
        # Further split the test data into validation and test sets
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=self.random_state)
        logging.info("Test data split into validation and test sets.")

        return train_data, val_data, test_data

    def get_data(self) -> Any:
        """
        Returns the loaded data.

        Returns:
        - Any: Loaded dataset.
        """
        return self.data