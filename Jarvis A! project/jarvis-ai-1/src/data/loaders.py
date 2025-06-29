from typing import Any, Dict
import pandas as pd
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    logging.info(f"Loading CSV file from {file_path}")
    return pd.read_csv(file_path)

def load_image(file_path: str) -> Any:
    """
    Load an image from a file.

    Parameters:
    - file_path (str): The path to the image file.

    Returns:
    - Any: The loaded image in a format suitable for processing.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    logging.info(f"Loading image from {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        logging.error(f"Failed to load image: {file_path}")
        raise ValueError(f"The file {file_path} is not a valid image.")
    
    return image