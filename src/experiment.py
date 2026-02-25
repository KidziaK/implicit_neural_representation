from dataclasses import dataclass
from torch import nn, optim, Tensor
from .loss.base import LossFunction
from .training_config import TrainingConfig
from .data import DataSampler


@dataclass
class Experiment:
    model: nn.Module
    loss_function: LossFunction
    data_sampler: DataSampler
    training_config: TrainingConfig
    optimizer: optim.Optimizer

    def train(self) -> None:
        self.model.train()

        for epoch in range(self.training_config.epochs):
            self.optimizer.zero_grad()

            data = self.data_sampler.sample()
            data.sdf(self.model)
            data.grad(self.model)

            loss = self.loss_function(data)
            loss.backward()

            self.optimizer.step()

    def evaluate(self, x: Tensor) -> Tensor:
        self.model.eval()
        flat_x = x.flatten()
        flat_sdf = self.model(flat_x)
        return flat_sdf.reshape(x.shape)
