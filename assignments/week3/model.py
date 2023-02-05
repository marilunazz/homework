import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    """
    A multilinear perceptron model that uses pytorch to classify.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: list,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.initializer = initializer
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()

        # adding list functionality to make it more than one hidden layer
        num_inputs = self.input_size
        next_num_inputs = 1
        length = 1
        if isinstance(self.hidden_size, list):
            length = len(self.hidden_size)
        for i in range(length):
            if isinstance(self.hidden_size, list):
                next_num_inputs = self.hidden_size[i]
            self.layers += [nn.Linear(num_inputs, next_num_inputs)]
            num_inputs = next_num_inputs
        self.out = nn.Linear(num_inputs, num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x_tensor = torch.tensor(x)
        for layer in self.layers:
            x = self.activation(layer(x_tensor))
        x = self.out(x)
        return x
