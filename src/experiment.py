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
            data.sdf(self.model)
            data.grad(self.model)

            loss = self.loss_function(data)
            y = "\033[33m"
            r = "\033[0m"
            loss_log_str = f"{y}{loss.item():.6f}{r}"

            loss.backward()

            self.optimizer.step()
            
            y = "\033[33m"
            r = "\033[0m"
            logger.info(f"Epoch {y}{epoch + 1}{r}/{self.training_config.epochs} - Loss: {loss_log_str}")


    def evaluate(self, x: Tensor) -> Tensor:
        self.model.eval()
        flat_x = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            flat_sdf = self.model(flat_x)
        return flat_sdf.reshape(x.shape[:-1])
