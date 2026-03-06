from .base import Norm
from torch import Tensor


def dirichlet_loss(y: Tensor, norm: Norm = Norm.L1):
    match norm:
        case Norm.L1:
            return y.abs().mean()
        case Norm.L2:
            return y.pow(2).mean()
