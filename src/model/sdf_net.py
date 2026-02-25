from torch import nn, Tensor
from .siren import SirenLayer

class SDFNet(nn.Module):
    def __init__(self, hidden_layers_num: int = 4, hidden_dim: int = 64, in_dims: int = 3, out_dims: int = 1):
        super().__init__()
        hidden_layers = [SirenLayer(hidden_dim, hidden_dim) for _ in range(hidden_layers_num)]

        self.backbone = nn.Sequential(
            SirenLayer(in_dims, hidden_dim, is_first=True),
            *hidden_layers
        )
        self.readout = nn.Linear(hidden_dim, out_dims)

    def forward(self, x: Tensor) -> Tensor:
        phi = self.backbone(x)
        return self.readout(phi)
