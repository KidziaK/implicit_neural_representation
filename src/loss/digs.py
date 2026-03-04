import torch
from torch import nn, Tensor
from .dirichlet import dirichlet_loss
from ..base.training_config import TrainingConfig
from .sample import sample_surface, sample_volume
from .eikonal import eikonal_loss_from_points_values
from .dnm import dnm_loss

def digs(model: nn.Module, config: TrainingConfig, surface_points: Tensor, t: float) -> Tensor:
    manifold_points = sample_surface(surface_points, config.manifold_points)
    volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds)

    x = torch.cat([manifold_points, volume_points], dim=0)
    y = model(x)

    y_manifold = y[:manifold_points.shape[0]]
    y_volume = y[manifold_points.shape[0]:]

    loss_weights = config.loss_weights

    loss_dirichlet = loss_weights.dirichlet(t) * dirichlet_loss(y_manifold)
    loss_dnm = loss_weights.dnm(t) * dnm_loss(y_volume, alpha=config.dnm_alpha)
    loss_eikonal = loss_weights.eikonal(t) * eikonal_loss_from_points_values(x, y)

    return loss_dirichlet + loss_eikonal + loss_dnm
