class Predictor:
    def __init__(self, model):
        """
        Initializes the Predictor with a trained model.

        Parameters:
        model: The trained model to be used for predictions.
        """
        self.model = model

    def predict(self, input_data):
        """
        Makes a prediction using the trained model.

        Parameters:
        input_data: The input data for which predictions are to be made.

        Returns:
        The predicted output from the model.
        """
        # Validate input data
        if not self._validate_input(input_data):
            raise ValueError("Invalid input data format.")

        prediction = self.model.predict(input_data)
        return prediction

    def _validate_input(self, input_data):
        """
        Validates the input data format.

        Parameters:
        input_data: The input data to validate.

        Returns:
        bool: True if input data is valid, False otherwise.
        """
        # Implement validation logic (e.g., check shape, type)
        return isinstance(input_data, (list, np.ndarray)) and len(input_data) > 0

    def load_model(self, model_path):
        """
        Loads a model from the specified path.

        Parameters:
        model_path: The path to the model file.

        Returns:
        The loaded model.
        """
        # Implement model loading logic (e.g., using joblib or pickle)
        import joblib
        self.model = joblib.load(model_path)