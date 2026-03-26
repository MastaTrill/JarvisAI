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
        Makes predictions on the provided input data.

        Parameters:
        input_data: The data on which predictions are to be made.

        Returns:
        The predictions made by the model.
        """
        # Validate input data
        if input_data is None or not isinstance(input_data, (list, np.ndarray)):
            raise ValueError("Input data must be a non-empty list or numpy array.")

        # Perform prediction
        predictions = self.model.predict(input_data)
        return predictions

    def load_model(self, model_path):
        """
        Loads a model from the specified file path.

        Parameters:
        model_path: The path to the model file.

        Returns:
        The loaded model.
        """
        import joblib
        self.model = joblib.load(model_path)

    def save_model(self, model_path):
        """
        Saves the current model to the specified file path.

        Parameters:
        model_path: The path where the model will be saved.
        """
        import joblib
        joblib.dump(self.model, model_path)