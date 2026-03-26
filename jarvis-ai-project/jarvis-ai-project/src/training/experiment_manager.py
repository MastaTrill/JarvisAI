class ExperimentManager:
    """Handles experiment tracking and configuration management."""

    def __init__(self, config: dict):
        """
        Initializes the ExperimentManager with the given configuration.

        Parameters:
            config (dict): A dictionary containing configuration settings for the experiment.
        """
        self.config = config
        self.experiment_id = None
        self.metrics = {}

    def start_experiment(self):
        """Starts a new experiment and generates a unique experiment ID."""
        # Logic to start an experiment and generate an ID
        self.experiment_id = "unique_experiment_id"  # Placeholder for actual ID generation
        self.log_experiment_start()

    def log_experiment_start(self):
        """Logs the start of the experiment."""
        logging.info(f"Experiment {self.experiment_id} started with config: {self.config}")

    def log_metric(self, metric_name: str, value: float):
        """
        Logs a metric for the current experiment.

        Parameters:
            metric_name (str): The name of the metric to log.
            value (float): The value of the metric.
        """
        self.metrics[metric_name] = value
        logging.info(f"Metric logged: {metric_name} = {value}")

    def end_experiment(self):
        """Ends the current experiment and logs the final metrics."""
        self.log_experiment_end()
        self.save_metrics()

    def log_experiment_end(self):
        """Logs the end of the experiment."""
        logging.info(f"Experiment {self.experiment_id} ended.")

    def save_metrics(self):
        """Saves the logged metrics to a persistent storage."""
        # Logic to save metrics (e.g., to a file or a database)
        logging.info(f"Metrics for experiment {self.experiment_id}: {self.metrics}")