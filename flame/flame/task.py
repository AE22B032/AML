"""Model and parameter helpers for the Flame app."""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Update model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)