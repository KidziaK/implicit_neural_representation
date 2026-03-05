import torch
from torch import nn, Tensor, autograd
from .dirichlet import dirichlet_loss
from .double_trace_gaussian_curvature import double_trace_loss
from ..base.training_config import TrainingConfig
from .sample import sample_surface, sample_volume
from .eikonal import eikonal_loss_from_grad
from .dnm import dnm_loss

def developable(model: nn.Module, config: TrainingConfig, surface_points: Tensor, t: float) -> dict[str, Tensor]:
    manifold_points = sample_surface(surface_points, config.manifold_points)
    volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=config.device)
    manifold_points.requires_grad_(True)

    x = torch.cat([manifold_points, volume_points], dim=0)
    x.requires_grad_(True)

    with torch.enable_grad():
        y = model(x)

    y_manifold = y[:manifold_points.shape[0]]
    y_volume = y[manifold_points.shape[0]:]

    grad_y = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    y_grad_manifold = grad_y[:manifold_points.shape[0]]

    loss_weights = config.loss_weights

    loss_dirichlet = loss_weights.dirichlet(t) * dirichlet_loss(y_manifold)
    loss_dnm = loss_weights.dnm(t) * dnm_loss(y_volume, alpha=config.dnm_alpha)
    loss_eikonal = loss_weights.eikonal(t) * eikonal_loss_from_grad(grad_y)
    loss_developable = loss_weights.developable(t) * double_trace_loss(manifold_points, y_grad_manifold)

    return {
        "loss_dirichlet": loss_dirichlet,
        "loss_dnm": loss_dnm,
        "loss_eikonal": loss_eikonal,
        "loss_developable": loss_developable
    }
