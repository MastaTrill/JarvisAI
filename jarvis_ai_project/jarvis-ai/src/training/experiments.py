from typing import Any, Dict
import logging
import json
import os

logger = logging.getLogger(__name__)

def log_experiment(experiment_name: str, parameters: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Logs the details of an experiment.

    Args:
        experiment_name (str): The name of the experiment.
        parameters (Dict[str, Any]): The parameters used for the experiment.
        metrics (Dict[str, Any]): The metrics obtained from the experiment.
    """
    experiment_data = {
        "experiment_name": experiment_name,
        "parameters": parameters,
        "metrics": metrics
    }
    
    log_file = os.path.join("logs", f"{experiment_name}.json")
    
    try:
        with open(log_file, 'w') as f:
            json.dump(experiment_data, f, indent=4)
        logger.info(f"Experiment '{experiment_name}' logged successfully.")
    except Exception as e:
        logger.error(f"Failed to log experiment '{experiment_name}': {e}")

def load_experiment(experiment_name: str) -> Dict[str, Any]:
    """Loads the details of a logged experiment.

    Args:
        experiment_name (str): The name of the experiment to load.

    Returns:
        Dict[str, Any]: The details of the experiment.
    """
    log_file = os.path.join("logs", f"{experiment_name}.json")
    
    if not os.path.exists(log_file):
        logger.error(f"Experiment '{experiment_name}' not found.")
        return {}
    
    try:
        with open(log_file, 'r') as f:
            experiment_data = json.load(f)
        logger.info(f"Experiment '{experiment_name}' loaded successfully.")
        return experiment_data
    except Exception as e:
        logger.error(f"Failed to load experiment '{experiment_name}': {e}")
        return {}