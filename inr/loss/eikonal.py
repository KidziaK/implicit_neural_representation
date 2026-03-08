import torch
from torch import Tensor, autograd
from .base import Norm


def eikonal_loss_from_grad(grad_y: Tensor, norm: Norm = Norm.L1):
    grad_y_norm = grad_y.norm(dim=-1)

    match norm:
        case Norm.L1:
            return (grad_y_norm - 1.0).abs().mean()
        case Norm.L2:
            return (grad_y_norm - 1.0).pow(2).mean()


def eikonal_loss_from_points_values(x: Tensor, y: Tensor, norm: Norm = Norm.L1):
    grad_y = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True,
    )[0]

    return eikonal_loss_from_grad(grad_y=grad_y, norm=norm)
