from torch import nn, optim, Tensor
from .base.training_config import TrainingConfig
from .base.loss import LossFunction
from dataclasses import dataclass
from time import time

@dataclass
class TrainingResult:
    training_time_s: float

def train(model: nn.Module, config: TrainingConfig, optimizer: optim.Optimizer, loss_function: LossFunction, surface_points: Tensor) -> TrainingResult:
    model.train()

    st = time()

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        t = epoch / config.epochs

        loss_dict = loss_function(model, config, surface_points, t)
        loss = Tensor(sum(l for l in loss_dict.values()))
        loss.backward()

        optimizer.step()

    ed = time()

    training_time_s = ed - st

    return TrainingResult(training_time_s=training_time_s)
