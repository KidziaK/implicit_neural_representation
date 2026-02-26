import torch.nn as nn
from torch import Tensor


class SDFNet(nn.Module):
    def __init__(self, hidden_layers_num: int = 4, hidden_dim: int = 64,
                 in_dims: int = 3, out_dims: int = 1):
        super().__init__()
        self.in_dims = in_dims

        hidden_layers = []
        for _ in range(hidden_layers_num):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())

        self.backbone = nn.Sequential(
            nn.Linear(in_dims, hidden_dim),
            nn.ReLU(),
            *hidden_layers
        )

        self.readout = nn.Linear(hidden_dim, out_dims)

    def forward(self, x: Tensor) -> Tensor:
        phi = self.backbone(x)
        return self.readout(phi)
