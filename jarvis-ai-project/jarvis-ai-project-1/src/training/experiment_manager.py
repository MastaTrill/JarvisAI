class ExperimentManager:
    def __init__(self, config: dict):
        """
        Initializes the ExperimentManager with the given configuration.

        Parameters:
        config (dict): A dictionary containing configuration settings for the experiment.
        """
        self.config = config
        self.experiment_id = None
        self.metrics = {}

    def start_experiment(self, experiment_name: str):
        """
        Starts a new experiment with the given name.

        Parameters:
        experiment_name (str): The name of the experiment to start.
        """
        self.experiment_id = experiment_name
        self.metrics = {}
        self.log(f"Experiment '{experiment_name}' started.")

    def log_metric(self, metric_name: str, value: float):
        """
        Logs a metric for the current experiment.

        Parameters:
        metric_name (str): The name of the metric to log.
        value (float): The value of the metric.
        """
        if self.experiment_id is not None:
            self.metrics[metric_name] = value
            self.log(f"Metric '{metric_name}' logged with value: {value}.")
        else:
            self.log("No experiment is currently running. Please start an experiment first.")

    def end_experiment(self):
        """
        Ends the current experiment and logs the metrics.
        """
        if self.experiment_id is not None:
            self.log(f"Experiment '{self.experiment_id}' ended.")
            self.log(f"Metrics: {self.metrics}")
            self.experiment_id = None
            self.metrics = {}
        else:
            self.log("No experiment is currently running.")

    def log(self, message: str):
        """
        Logs a message to the console.

        Parameters:
        message (str): The message to log.
        """
        print(message)  # Replace with a logging framework as needed