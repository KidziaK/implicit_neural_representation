import torch
import numpy as np
from torch import nn, Tensor
from enum import Enum

class ActivationType(Enum):
    RELU = 1
    SOFTPLUS = 2
    SIREN = 3

class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SDFNet(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        hidden_layers: int = 4,
        hidden_dim: int = 256,
        activation_type: ActivationType = ActivationType.SIREN
    ) -> None:
        super().__init__()
        self.activation_type = activation_type

        layers = []

        match activation_type:
            case ActivationType.SIREN:
                layers.append(SineLayer(in_features, hidden_dim, is_first=True, omega_0=1.0))
                layers += [SineLayer(hidden_dim, hidden_dim, is_first=False) for _ in range(hidden_layers)]
            case ActivationType.SOFTPLUS:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.Softplus(beta=100))
                for _ in range(hidden_layers):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.Softplus(beta=100))
            case ActivationType.RELU:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.ReLU())
                for _ in range(hidden_layers):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
