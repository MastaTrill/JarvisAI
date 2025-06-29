from typing import Any, Dict
import pandas as pd
import os

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_json(file_path)

def load_excel(file_path: str, sheet_name: str = 0) -> pd.DataFrame:
    """Load an Excel file into a pandas DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The name or index of the sheet to load. Defaults to 0.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_excel(file_path, sheet_name=sheet_name)