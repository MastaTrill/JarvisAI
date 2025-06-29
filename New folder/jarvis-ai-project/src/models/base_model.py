class BaseModel:
    """
    Base class for machine learning models.

    This class provides a template for training and evaluation methods that can be 
    extended by specific model implementations.

    Attributes:
        model: The underlying model architecture.
    """

    def __init__(self):
        self.model = None

    def train(self, train_data, train_labels):
        """
        Train the model on the provided training data.

        Args:
            train_data: The input data for training.
            train_labels: The corresponding labels for the training data.
        """
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model on the provided test data.

        Args:
            test_data: The input data for evaluation.
            test_labels: The corresponding labels for the test data.

        Returns:
            A dictionary containing evaluation metrics.
        """
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")