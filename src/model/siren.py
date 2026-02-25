import numpy as np
from torch import nn, sin , no_grad, Tensor

class SirenLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
        self.omega = 30.0

        with no_grad():
            bound = 1 / in_f if is_first else np.sqrt(6 / in_f) / self.omega
            self.lin.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return sin(self.omega * self.lin(x))
