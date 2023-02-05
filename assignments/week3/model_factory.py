import torch
from model import MLP


def create_model(input_dim: int, output_dim: int, hidden_dims: list = [32]) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.
            model = create_model(784, 10)

    """
    hidden_dims = [128]
    return MLP(
        input_dim, hidden_dims, output_dim, 1, torch.nn.ReLU, torch.nn.init.ones_
    )
