from typing import Any, Dict
import yaml
import json
import os

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def load_config(file_path: str) -> Dict[str, Any]:
    """Load a configuration file (YAML or JSON).

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, ext = os.path.splitext(file_path)
    if ext == '.yaml' or ext == '.yml':
        return load_yaml_config(file_path)
    elif ext == '.json':
        return load_json_config(file_path)
    else:
        raise ValueError("Unsupported file extension. Use .yaml or .json")