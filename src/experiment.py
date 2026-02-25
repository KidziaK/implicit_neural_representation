import torch
from dataclasses import dataclass
from torch import nn, optim, Tensor
from .loss.base import LossFunction
from .training_config import TrainingConfig
from .data import DataSampler
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class Experiment:
    model: nn.Module
    loss_function: LossFunction
    data_sampler: DataSampler
    training_config: TrainingConfig
    optimizer: optim.Optimizer

    def train(self) -> None:
        self.model.train()
        logger.info(f"Starting training with loss function: {self.loss_function}")

        for epoch in range(self.training_config.epochs):
            self.optimizer.zero_grad()

            data = self.data_sampler.sample()
            data.grad(self.model)

            loss_dict = self.loss_function(data)
            loss = torch.stack(list(loss_dict.values())).sum(dim=0)

            y = "\033[33m"
            r = "\033[0m"

            loss_list = [f"{name}: {y}{tensor.item():.4f}{r}" for name, tensor in loss_dict.items()]
            lost_str = ", ".join(loss_list)
            lost_str += f", total: {y}{loss.item():.4f}{r}"

            loss.backward()

            self.optimizer.step()

            logger.info(f"Epoch {y}{epoch + 1}{r}/{self.training_config.epochs} - {lost_str}")


    def evaluate(self, x: Tensor) -> Tensor:
        self.model.eval()
        flat_x = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            flat_sdf = self.model(flat_x)
        return flat_sdf.reshape(x.shape[:-1])
