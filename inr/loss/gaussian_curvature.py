import math
import torch
from torch import Tensor
from .base import Norm


def double_trough_curve(curvature: Tensor) -> Tensor:
    pi = math.pi
    t = curvature.abs()

    a = (64 * pi - 80) / (pi**4)
    b = -(64 * pi - 88) / (pi**3)
    c = (16 * pi - 29) / (pi**2)
    d = 3 / pi

    return a * (t**4) + b * (t**3) + c * (t**2) + d * t


def gaussian_curvature_loss(
    x: Tensor,
    y: Tensor,
    eps: float = 1e-12,
    norm: Norm = Norm.L1,
) -> Tensor:
    grad_y = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]

    H_rows = []
    for i in range(3):
        grad_y_i = grad_y[..., i : i + 1]
        H_i = torch.autograd.grad(
            outputs=grad_y_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_y_i),
            create_graph=True,
            retain_graph=True,
        )[0]
        H_rows.append(H_i)

    H = torch.stack(H_rows, dim=-1)

    extended_matrix = torch.empty(grad_y.shape[:-1] + (4, 4), device=x.device, dtype=x.dtype)
    extended_matrix[..., :3, :3] = H
    extended_matrix[..., :3, 3] = grad_y
    extended_matrix[..., 3, :3] = grad_y
    extended_matrix[..., 3, 3] = 0.0

    det = torch.linalg.det(extended_matrix)

    match norm:
        case Norm.L1:
            grad_norm = torch.norm(grad_y, dim=-1, keepdim=True)
        case Norm.L2:
            grad_norm = torch.norm(grad_y, dim=-1, keepdim=True).pow(2)

    K = (-1.0 / (grad_norm + eps)) * det

    loss_vals = double_trough_curve(K)
    return loss_vals.mean()
