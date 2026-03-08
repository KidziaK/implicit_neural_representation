import torch
from torch import nn, optim, Tensor, tensor

from .loss.dirichlet import dirichlet_loss
from .loss.dnm import dnm_loss
from .loss.eikonal import eikonal_loss_from_grad, eikonal_loss_from_points_values
from .loss.gaussian_curvature import gaussian_curvature_loss
from .settings import get_device
from .training_config import TrainingConfig
from inr.sample import compute_sigmas, sample_volume, sample_near_surface
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

    sigmas = compute_sigmas(surface_points)
    weights = config.loss_weights
    surface_points.requires_grad_()

    st = time()

    logger.info("Training started")

    for epoch in range(config.epochs + 1):
        optimizer.zero_grad()

        t = epoch / config.epochs

        volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=get_device())
        near_surface_points = sample_near_surface(surface_points, sigmas)

        volume_points.requires_grad_()
        near_surface_points.requires_grad_()

        loss = tensor([0.0], device=get_device())

        y_manifold = model(surface_points)
        y_volume = model(volume_points)
        y_near = model(near_surface_points)

        if weights.dirichlet:
            loss_dirichlet = dirichlet_loss(y_manifold)
            loss += weights.dirichlet(t) * loss_dirichlet

        if weights.dnm:
            loss_dnm = dnm_loss(y_volume)
            loss += weights.dnm(t) * loss_dnm

        if weights.eikonal:
            loss_eikonal_manifold = eikonal_loss_from_points_values(surface_points, y_manifold)
            loss_eikonal_volume = eikonal_loss_from_points_values(volume_points, y_volume)
            loss += weights.eikonal(t) * (loss_eikonal_manifold + loss_eikonal_volume)

        if weights.gaussian_curvature:
            loss_gaussian_curvature = gaussian_curvature_loss(near_surface_points, y_near)
            loss += weights.gaussian_curvature(t) * loss_gaussian_curvature

        loss.backward()

        optimizer.step()

        logger.info(f"Epoch [{epoch}/{config.epochs}] | Loss {loss.item():.4f}")

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s)
