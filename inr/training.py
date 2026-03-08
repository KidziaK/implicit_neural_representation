import torch
from torch import nn, optim, Tensor, tensor, autograd
from .loss.dirichlet import dirichlet_loss
from .loss.dnm import dnm_loss
from .loss.eikonal import eikonal_loss_from_points_values, eikonal_loss_from_grad
from .loss.gauss_bonnet import gauss_bonnet_loss
from .loss.ncadr import ncadr_gaussian_curvature_loss
from .loss.flatcad import flatcad_loss
from .settings import get_device
from .training_config import TrainingConfig, eval_weight
from inr.sample import compute_sigmas, sample_volume, sample_near_surface
from loguru import logger
from time import time
from collections import defaultdict
from pydantic import BaseModel


class TrainingResult(BaseModel):
    training_time_s: float
    loss_dict: dict[str, float]


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
    loss_dict = defaultdict(lambda: 0.0)
    n = surface_points.size(0)

    st = time()

    logger.info("Training started")

    for epoch in range(config.epochs + 1):
        optimizer.zero_grad()

        t = epoch / config.epochs

        volume_points = sample_volume(n=config.volume_points, bounds=config.volume_bounds, device=get_device())
        volume_points.requires_grad_()

        near_points = sample_near_surface(surface_points, sigmas)
        near_points.requires_grad_()

        y_surface = model(surface_points)
        y_near = model(near_points)
        y_volume = model(volume_points)

        grad_y_surface = autograd.grad(
            outputs=y_surface,
            inputs=surface_points,
            grad_outputs=torch.ones_like(y_surface),
            create_graph=True,
            allow_unused=True,
        )[0]

        grad_y_volume = autograd.grad(
            outputs=y_volume,
            inputs=volume_points,
            grad_outputs=torch.ones_like(y_volume),
            create_graph=True,
            allow_unused=True,
        )[0]

        grad_y_near = autograd.grad(
            outputs=y_near,
            inputs=near_points,
            grad_outputs=torch.ones_like(y_near),
            create_graph=True,
            allow_unused=True,
        )[0]

        loss = tensor(0.0, device=get_device())

        if weights.dirichlet:
            loss_dirichlet = eval_weight(weights.dirichlet, t) * dirichlet_loss(y_surface)
            loss += loss_dirichlet
            loss_dict["dirichlet"] = loss_dirichlet

        if weights.dnm:
            loss_dnm = eval_weight(weights.dnm, t) * dnm_loss(y_volume)
            loss += loss_dnm
            loss_dict["dnm"] = loss_dnm.item()

        if weights.eikonal:
            total_grad = torch.cat([grad_y_surface, grad_y_volume, grad_y_near, grad_y_near], dim=0)
            loss_eikonal = eval_weight(weights.eikonal, t) * eikonal_loss_from_grad(total_grad)
            loss += loss_eikonal
            loss_dict["eikonal"] = loss_eikonal.item()

        if weights.gauss_bonnet:
            loss_gauss_bonnet_surface = gauss_bonnet_loss(surface_points, grad_y_surface)
            loss_gauss_bonnet_near = gauss_bonnet_loss(near_points, grad_y_near)
            loss_gauss_bonnet = eval_weight(weights.gauss_bonnet, t) * 0.5 *  (loss_gauss_bonnet_near + loss_gauss_bonnet_surface)
            loss += loss_gauss_bonnet
            loss_dict["gauss_bonnet"] = loss_gauss_bonnet.item()

        if weights.ncadr:
            loss_ncadr_near = ncadr_gaussian_curvature_loss(near_points, grad_y_near)
            loss_ncadr_surface = ncadr_gaussian_curvature_loss(surface_points, grad_y_surface)
            loss_ncadr = eval_weight(weights.ncadr, t) * 0.5 * (loss_ncadr_near + loss_ncadr_surface)
            loss += loss_ncadr
            loss_dict["ncadr"] = loss_ncadr.item()

        if weights.flatcad:
            loss_flatcad_near = flatcad_loss(near_points, grad_y_near)
            loss_flatcad_surface = flatcad_loss(surface_points, grad_y_surface)
            loss_flatcad = eval_weight(weights.flatcad, t) * 0.5 * (loss_flatcad_near + loss_flatcad_surface)
            loss += loss_flatcad
            loss_dict["flatcad"] = loss_flatcad.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        if epoch % 50 == 0:
            logger.info(f"Epoch [{epoch}/{config.epochs}] | Loss {loss.item():.4f}")

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s, loss_dict=loss_dict)
