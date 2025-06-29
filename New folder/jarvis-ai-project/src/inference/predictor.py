class Predictor:
    def __init__(self, model):
        """
        Initializes the Predictor with a trained model.

        Parameters:
        model: The trained machine learning model to be used for predictions.
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
        if input_data is None:
            raise ValueError("Input data cannot be None.")
        
        # Make predictions
        predictions = self.model.predict(input_data)
        return predictions

    def batch_predict(self, batch_data):
        """
        Makes predictions on a batch of input data.

        Parameters:
        batch_data: A list or array of data on which predictions are to be made.

        Returns:
        A list of predictions made by the model for each input in the batch.
        """
        if not isinstance(batch_data, (list, tuple)):
            raise ValueError("Batch data must be a list or tuple.")
        
        predictions = [self.predict(data) for data in batch_data]
        return predictions