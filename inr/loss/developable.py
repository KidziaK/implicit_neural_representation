import torch
from torch import nn, Tensor, autograd
from .dirichlet import dirichlet_loss
from .double_trace_gaussian_curvature import double_trace_loss
from ..training_config import TrainingConfig
from .sample import sample_volume, sample_near_surface
from .eikonal import eikonal_loss_from_grad
from .dnm import dnm_loss


def developable_loss(model: nn.Module, config: TrainingConfig, surface_points: Tensor, t: float) -> dict[str, Tensor]:
    volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=config.device)
    n_near = surface_points.shape[0]
    surface_for_near = surface_points[torch.randperm(surface_points.shape[0], device=surface_points.device)[:n_near]]
    near_points = sample_near_surface(surface_for_near)

    n_surface = surface_points.shape[0]

    surface_and_near = torch.cat([surface_points, near_points], dim=0).requires_grad_(True)
    x = torch.cat([surface_and_near, volume_points], dim=0)
    x.requires_grad_(True)

    with torch.enable_grad():
        y = model(x)

    y_manifold = y[:n_surface]
    y_volume = y[n_surface + n_near :]

    grad_y = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]

    grad_for_developable = grad_y[: n_surface + n_near]

    loss_weights = config.loss_weights

    loss_dirichlet = loss_weights.dirichlet(t) * dirichlet_loss(y_manifold)
    loss_dnm = loss_weights.dnm(t) * dnm_loss(y_volume, alpha=config.dnm_alpha)
    loss_eikonal = loss_weights.eikonal(t) * eikonal_loss_from_grad(grad_y)
    loss_developable = loss_weights.developable(t) * double_trace_loss(surface_and_near, grad_for_developable)

    return {
        "loss_dirichlet": loss_dirichlet,
        "loss_dnm": loss_dnm,
        "loss_eikonal": loss_eikonal,
        "loss_developable": loss_developable,
    }
