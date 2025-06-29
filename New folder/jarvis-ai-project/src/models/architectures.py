from typing import Any, Dict
import torch
import torch.nn as nn

class SimpleFeedForwardNN(nn.Module):
    """
    A simple feedforward neural network architecture.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output classes.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initializes the SimpleFeedForwardNN with the given parameters.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
        """
        super(SimpleFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvolutionalNN(nn.Module):
    """
    A convolutional neural network architecture.

    Attributes:
        num_classes (int): The number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initializes the ConvolutionalNN with the given number of classes.

        Args:
            num_classes (int): The number of output classes.
        """
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name: str, **kwargs: Any) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): The name of the model to retrieve.
        **kwargs: Additional parameters for model initialization.

    Returns:
        nn.Module: The requested model architecture.
    """
    if model_name == "simple_ffnn":
        return SimpleFeedForwardNN(**kwargs)
    elif model_name == "conv_nn":
        return ConvolutionalNN(**kwargs)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")