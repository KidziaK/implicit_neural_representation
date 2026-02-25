from typing import Protocol
from torch import nn

class TrainingData(Protocol):
    def grad(self, model: nn.Module) -> None: ...

class DataSampler(Protocol):
    def sample(self) -> TrainingData: ...
