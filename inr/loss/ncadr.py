import math
import torch
from torch import Tensor, nn
from ..base.training_config import TrainingConfig
from .sample import sample_volume, sample_near_surface
from .dirichlet import dirichlet_loss
from .dnm import dnm_loss
from .eikonal import eikonal_loss_from_points_values


def ncr_linear_decay(t: float) -> float:
    if t < 0.2:
        return 10.0
    if t < 0.5:
        return 10.0 + (0.001 - 10.0) * (t - 0.2) / (0.5 - 0.2)
    if t < 1.0:
        return 0.001 + (0.0 - 0.001) * (t - 0.5) / (1.0 - 0.5)
    return 0.0


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
    min_grad_norm_sq: float = 0.01,
    max_abs_k: float = 10.0,
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

    grad_norm2 = grad_y.norm(dim=-1) ** 2
    grad_norm2_safe = grad_norm2.clamp(min=min_grad_norm_sq)
    K = (-1.0 / (grad_norm2_safe + eps)) * det
    K = K.clamp(min=-max_abs_k, max=max_abs_k)

    loss_vals = double_trough_curve(K)
    return loss_vals.mean()


def ncadr_loss(model: nn.Module, config: TrainingConfig, surface_points: Tensor, t: float) -> dict[str, Tensor]:
    volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=config.device)
    n_near = surface_points.shape[0]
    surface_for_near = surface_points[torch.randperm(surface_points.shape[0], device=surface_points.device)[:n_near]]
    near_points = sample_near_surface(surface_for_near)

    surface_points = surface_points.requires_grad_(True)
    near_points.requires_grad_(True)

    x_for_eikonal = torch.cat([surface_points, near_points], dim=0)
    x_for_eikonal.requires_grad_(True)

    with torch.enable_grad():
        y_manifold = model(surface_points)
        y_volume = model(volume_points)
        y_near = model(near_points)

        y_for_eikonal = model(x_for_eikonal)

    loss_weights = config.loss_weights

    loss_dirichlet = loss_weights.dirichlet(t) * dirichlet_loss(y_manifold)
    loss_dnm = loss_weights.dnm(t) * dnm_loss(y_volume, alpha=config.dnm_alpha)
    loss_eikonal = loss_weights.eikonal(t) * eikonal_loss_from_points_values(x_for_eikonal, y_for_eikonal)

    loss_near = gaussian_curvature_loss(near_points, y_near)
    if config.bidirectional_ncr:
        loss_manifold = gaussian_curvature_loss(surface_points, y_manifold)
        loss_ncadr = loss_weights.ncr(t) * (0.5 * loss_near + 0.5 * loss_manifold)
    else:
        loss_ncadr = loss_weights.ncr(t) * loss_near

    return {
        "loss_dirichlet": loss_dirichlet,
        "loss_dnm": loss_dnm,
        "loss_eikonal": loss_eikonal,
        "loss_ncr": loss_ncadr,
    }
