from typing import Callable
from torch import Tensor, nn
from .training_config import TrainingConfig

LossFunction = Callable[[nn.Module, TrainingConfig, Tensor, float], dict[str, Tensor]]
