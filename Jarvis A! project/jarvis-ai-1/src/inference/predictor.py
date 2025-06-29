from typing import Any, Dict
import joblib
import numpy as np
import logging

class Predictor:
    """Class for handling model inference."""

    def __init__(self, model_path: str) -> None:
        """
        Initializes the Predictor with the path to the trained model.

        Args:
            model_path (str): Path to the trained model file.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self) -> Any:
        """
        Loads the trained model from the specified path.

        Returns:
            Any: The loaded model.
        """
        try:
            model = joblib.load(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the loaded model.

        Args:
            input_data (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted output.
        """
        try:
            predictions = self.model.predict(input_data)
            logging.info("Predictions made successfully.")
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise

    def predict_proba(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes probability predictions using the loaded model.

        Args:
            input_data (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        try:
            probabilities = self.model.predict_proba(input_data)
            logging.info("Probability predictions made successfully.")
            return probabilities
        except Exception as e:
            logging.error(f"Error making probability predictions: {e}")
            raise