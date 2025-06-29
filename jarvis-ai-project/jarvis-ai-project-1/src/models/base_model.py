class BaseModel:
    def __init__(self):
        """Initialize the base model."""
        pass

    def train(self, data, labels):
        """
        Train the model on the provided data.

        Parameters:
        data (array-like): The input data for training.
        labels (array-like): The corresponding labels for the input data.
        """
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def evaluate(self, data, labels):
        """
        Evaluate the model on the provided data.

        Parameters:
        data (array-like): The input data for evaluation.
        labels (array-like): The corresponding labels for the input data.

        Returns:
        float: The evaluation metric (e.g., accuracy).
        """
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")