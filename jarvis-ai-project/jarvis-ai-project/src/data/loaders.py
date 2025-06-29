from typing import Any, Dict
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a specified file path.

    Parameters:
    - file_path (str): The path to the dataset file.

    Returns:
    - pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - ValueError: If the file format is not supported.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        logging.info(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        logging.info(f"Loading Excel file: {file_path}")
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        logging.info(f"Loading JSON file: {file_path}")
        return pd.read_json(file_path)
    else:
        logging.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")