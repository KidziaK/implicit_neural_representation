import torch
from .base import LossFunction
from ..data import SimpleTrainingData
from torch import Tensor


class EikonalLoss(LossFunction):
    def __call__(self, x: SimpleTrainingData) -> Tensor:
        all_grads = torch.concat([x.on_manifold_points.grad, x.off_manifold_points.grad])
        return (all_grads.norm(dim=-1) - 1).mean()
