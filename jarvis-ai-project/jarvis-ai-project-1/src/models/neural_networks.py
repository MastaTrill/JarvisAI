from typing import Any, Dict, Tuple
import tensorflow as tf
from .base_model import BaseModel

class NeuralNetwork(BaseModel):
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train: Any, y_train: Any, epochs: int, batch_size: int, validation_data: Tuple[Any, Any]) -> None:
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x_test: Any, y_test: Any) -> Dict[str, float]:
        return self.model.evaluate(x_test, y_test, verbose=0)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)