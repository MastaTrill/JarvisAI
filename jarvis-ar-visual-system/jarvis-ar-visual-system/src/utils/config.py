import os
import yaml
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)

def load_config(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    if not os.path.exists(file_path):
        logging.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {file_path}")
        return config

def get_rendering_config() -> Dict[str, Any]:
    """Get the rendering configuration.

    Returns:
        Dict[str, Any]: The rendering configuration.
    """
    return load_config(os.path.join('config', 'rendering_config.yaml'))

def get_ar_config() -> Dict[str, Any]:
    """Get the augmented reality configuration.

    Returns:
        Dict[str, Any]: The augmented reality configuration.
    """
    return load_config(os.path.join('config', 'ar_config.yaml'))