import torch
from torch import Tensor


def dnm_loss(y: Tensor, alpha: float = 100.0) -> Tensor:
    return torch.exp(-alpha * y.abs()).mean()
