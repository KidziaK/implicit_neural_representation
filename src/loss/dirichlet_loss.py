import torch
from ..data import SimpleTrainingData
from torch import Tensor

def dirichlet_loss(x: SimpleTrainingData) -> Tensor:
    return x.on_manifold_points_sdf.abs().mean()
