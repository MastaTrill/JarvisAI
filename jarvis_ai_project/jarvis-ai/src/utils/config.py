from typing import Any, Dict
import yaml
import os

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The configuration file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_config(model_config_path: str, training_config_path: str) -> Dict[str, Any]:
    """Get the combined configuration from model and training YAML files.

    Args:
        model_config_path (str): The path to the model configuration file.
        training_config_path (str): The path to the training configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the combined configuration.
    """
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    
    return {
        'model': model_config,
        'training': training_config
    }