import torch
from ..data import SimpleTrainingData
from torch import Tensor

def eikonal_loss_l1(x: SimpleTrainingData) -> Tensor:
    all_grads = torch.concat([x.on_manifold_points_grad, x.off_manifold_points_grad])
    return (all_grads.norm(dim=-1) - 1).abs().mean()

def eikonal_loss_l2(x: SimpleTrainingData) -> Tensor:
    all_grads = torch.concat([x.on_manifold_points_grad, x.off_manifold_points_grad])
    return (all_grads.norm(dim=-1) - 1).pow(2).mean()
