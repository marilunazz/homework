import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
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
        #self.activation = activation
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(input_size, hidden_size)]  
        self.out = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # print(' input dim: ', self.input_size)
        # print('hidden size: ', self.hidden_size)
        # print('out: ', self.num_classes)
        x_tensor = torch.tensor(x)
        for layer in self.layers:
            l = layer(x_tensor)
            # print(l.shape, ' l shape')
            x = self.activation(l)
        x = self.out(x)
        return x
