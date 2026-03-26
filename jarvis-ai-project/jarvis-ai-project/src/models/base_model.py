from abc import ABC, abstractmethod
import logging

class BaseModel(ABC):
    """
    Base class for all models in the Jarvis AI project.

    Attributes:
        model: The underlying model architecture.
    """

    def __init__(self):
        """
        Initializes the BaseModel instance.
        """
        self.model = None
        logging.info("BaseModel initialized.")

    @abstractmethod
    def build_model(self):
        """
        Abstract method to build the model architecture.
        This method should be implemented by subclasses.
        """
        pass

    def train(self, training_data, validation_data, epochs: int):
        """
        Trains the model on the provided training data.

        Parameters:
            training_data: The data to train the model on.
            validation_data: The data to validate the model during training.
            epochs (int): The number of epochs to train the model.

        Returns:
            None
        """
        logging.info(f"Starting training for {epochs} epochs.")
        # Implement training logic here
        pass

    def evaluate(self, test_data):
        """
        Evaluates the model on the provided test data.

        Parameters:
            test_data: The data to evaluate the model on.

        Returns:
            metrics: Evaluation metrics for the model.
        """
        logging.info("Evaluating model.")
        # Implement evaluation logic here
        pass