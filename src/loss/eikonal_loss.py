import torch
from ..data import SimpleTrainingData
from torch import Tensor

def eikonal_loss_l1(data: SimpleTrainingData) -> Tensor:
    all_grads = torch.concat([data.on_manifold_points_grad, data.off_manifold_points_grad])
    return (all_grads.norm(dim=-1) - 1).abs().mean()

def eikonal_loss_l2(data: SimpleTrainingData) -> Tensor:
    all_grads = torch.concat([data.on_manifold_points_grad, data.off_manifold_points_grad])
    return (all_grads.norm(dim=-1) - 1).pow(2).mean()
