import torch
from torch import nn, optim, Tensor, tensor

from .loss.dirichlet import dirichlet_loss
from .loss.dnm import dnm_loss
from .loss.eikonal import eikonal_loss_from_points_values
from .settings import get_device
from .training_config import TrainingConfig
from inr.sample import compute_sigmas, sample_volume
from dataclasses import dataclass
from loguru import logger
from time import time


@dataclass
class TrainingResult:
    training_time_s: float


def train(
    model: nn.Module,
    config: TrainingConfig,
    optimizer: optim.Optimizer,
    surface_points: Tensor,
) -> TrainingResult:
    model.train()

    sigmas = compute_sigmas(surface_points, k=51)
    weights = config.loss_weights
    surface_points.requires_grad_()

    st = time()

    logger.info("Training started")

    for epoch in range(config.epochs + 1):
        optimizer.zero_grad()

        t = epoch / config.epochs

        volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=get_device())
        volume_points.requires_grad_()

        x = torch.cat((surface_points, volume_points), dim=-2)
        y = model(x)

        y_surface = y[:surface_points.size(0)]
        y_volume = y[surface_points.size(0):]

        loss = tensor(0.0, device=get_device())

        if weights.dirichlet:
            loss_dirichlet = dirichlet_loss(y_surface)
            loss += weights.dirichlet(t) * loss_dirichlet

        if weights.dnm:
            loss_dnm = dnm_loss(y_volume)
            loss += weights.dnm(t) * loss_dnm

        if weights.eikonal:
            x.requires_grad_()
            y.requires_grad_()
            loss += weights.eikonal(t) * eikonal_loss_from_points_values(x, y)

        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:
            logger.info(f"Epoch [{epoch}/{config.epochs}] | Loss {loss.item():.4f}")

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s)
