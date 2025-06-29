from typing import Any, Dict
import os
import logging
import json

class ExperimentManager:
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initializes the ExperimentManager.

        Parameters:
        - experiment_name (str): The name of the experiment.
        - log_dir (str): Directory to save logs and checkpoints.
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        """
        Sets up logging for the experiment.
        """
        logging.basicConfig(
            filename=os.path.join(self.log_dir, f"{self.experiment_name}.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Experiment '{self.experiment_name}' started.")

    def save_checkpoint(self, model: Any, epoch: int, metrics: Dict[str, Any]):
        """
        Saves the model checkpoint.

        Parameters:
        - model (Any): The model to save.
        - epoch (int): The current epoch number.
        - metrics (Dict[str, Any]): Metrics to log with the checkpoint.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        # Assuming model has a method to save itself
        model.save(checkpoint_path)
        logging.info(f"Checkpoint saved at '{checkpoint_path}' with metrics: {metrics}")

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """
        Logs metrics for the current epoch.

        Parameters:
        - epoch (int): The current epoch number.
        - metrics (Dict[str, Any]): Metrics to log.
        """
        metrics_log_path = os.path.join(self.checkpoint_dir, "metrics.json")
        if os.path.exists(metrics_log_path):
            with open(metrics_log_path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics[epoch] = metrics

        with open(metrics_log_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        logging.info(f"Metrics logged for epoch {epoch}: {metrics}")