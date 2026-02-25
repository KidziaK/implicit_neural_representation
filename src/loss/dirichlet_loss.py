import torch
from ..data import SimpleTrainingData
from torch import Tensor

def dirichlet_on_manifold_loss(data: SimpleTrainingData) -> Tensor:
    return data.on_manifold_points_sdf.abs().mean()

def dirichlet_off_manifold_loss(data: SimpleTrainingData) -> Tensor:
    alpha = 100.
    return torch.exp(-alpha * data.off_manifold_points_sdf.abs()).mean()
