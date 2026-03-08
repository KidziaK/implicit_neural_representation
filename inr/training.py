import torch
from torch import nn, optim, Tensor, tensor, autograd
from .loss.dirichlet import dirichlet_loss
from .loss.dnm import dnm_loss
from .loss.eikonal import eikonal_loss_from_points_values, eikonal_loss_from_grad
from .loss.gauss_bonnet import gauss_bonnet_loss
from .settings import get_device
from .training_config import TrainingConfig
from inr.sample import compute_sigmas, sample_volume
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

        x.requires_grad_()
        y.requires_grad_()

        grad_y = autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            allow_unused=True,
        )[0]

        loss = tensor(0.0, device=get_device())

        if weights.dirichlet:
            loss_dirichlet = weights.dirichlet(t) * dirichlet_loss(y_surface)
            loss += loss_dirichlet
            loss_dict["dirichlet"] = loss_dirichlet

        if weights.dnm:
            loss_dnm = weights.dnm(t) * dnm_loss(y_volume)
            loss += loss_dnm
            loss_dict["dnm"] = loss_dnm.item()

        if weights.eikonal:
            loss_eikonal =  weights.eikonal(t) * eikonal_loss_from_grad(grad_y)
            loss += loss_eikonal
            loss_dict["eikonal"] = loss_eikonal.item()

        if weights.gauss_bonnet:
            loss_gauss_bonnet = weights.gauss_bonnet(t) * gauss_bonnet_loss(x, grad_y)
            loss += loss_gauss_bonnet
            loss_dict["gauss_bonnet"] = loss_gauss_bonnet.item()


        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:
            logger.info(f"Epoch [{epoch}/{config.epochs}] | Loss {loss.item():.4f}")

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s, loss_dict=loss_dict)
