from torch import nn, optim, Tensor
from .base.training_config import TrainingConfig
from .base.loss import LossFunction
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
    loss_function: LossFunction,
    surface_points: Tensor,
) -> TrainingResult:
    model.train()

    st = time()

    logger.info("Training started")

    for epoch in range(config.epochs + 1):
        optimizer.zero_grad()

        t = epoch / config.epochs

        loss_dict = loss_function(model, config, surface_points, t)
        loss = Tensor(sum(l for l in loss_dict.values()))
        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            logger.info(
                f"Epoch [{epoch}/{config.epochs}] | Loss {loss.item():.4f}",
                loss_dict=loss_dict,
            )

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s)
