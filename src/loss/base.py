from typing import Callable, TypeVar
from ..data import TrainingData
from torch import Tensor
from dataclasses import dataclass

FunctionName = TypeVar("FunctionName", bound=str)

@dataclass
class LossFunction:
    weights: list[float]
    losses: list[Callable]

    def __call__(self, data: TrainingData) -> dict[FunctionName, Tensor]:
        return {
            loss.__name__: w * loss(data)
            for loss, w in zip(self.losses, self.weights)
        }
