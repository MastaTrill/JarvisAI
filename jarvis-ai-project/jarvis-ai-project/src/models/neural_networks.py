from typing import Any, Dict
import tensorflow as tf
from src.models.base_model import BaseModel
import logging

logging.basicConfig(level=logging.INFO)

class NeuralNetwork(BaseModel):
    def __init__(self, input_shape: tuple, num_classes: int, **kwargs: Any) -> None:
        """
        Initializes the NeuralNetwork model.

        Parameters:
        - input_shape (tuple): Shape of the input data.
        - num_classes (int): Number of output classes.
        - kwargs (Any): Additional keyword arguments for model configuration.
        """
        super().__init__(**kwargs)
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape: tuple, num_classes: int) -> tf.keras.Model:
        """
        Builds the neural network architecture.

        Parameters:
        - input_shape (tuple): Shape of the input data.
        - num_classes (int): Number of output classes.

        Returns:
        - tf.keras.Model: Compiled Keras model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("Model built successfully.")
        return model

    def train(self, x_train: Any, y_train: Any, epochs: int, batch_size: int) -> None:
        """
        Trains the neural network model.

        Parameters:
        - x_train (Any): Training data.
        - y_train (Any): Training labels.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Size of the training batches.
        """
        logging.info("Starting training...")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        logging.info("Training completed.")

    def evaluate(self, x_test: Any, y_test: Any) -> Dict[str, float]:
        """
        Evaluates the model on the test data.

        Parameters:
        - x_test (Any): Test data.
        - y_test (Any): Test labels.

        Returns:
        - Dict[str, float]: Evaluation metrics.
        """
        logging.info("Evaluating model...")
        metrics = self.model.evaluate(x_test, y_test)
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(self, x: Any) -> Any:
        """
        Makes predictions using the trained model.

        Parameters:
        - x (Any): Input data for predictions.

        Returns:
        - Any: Predicted classes.
        """
        logging.info("Making predictions...")
        predictions = self.model.predict(x)
        return predictions.argmax(axis=1)