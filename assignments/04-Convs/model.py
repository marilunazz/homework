import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A CNN with XXX and y to classify images quickly
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_channels, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2352, 120)  # used to be 16*5*5, 120
        self.fc2 = nn.Linear(120, self.num_classes)
        self.drop = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x
