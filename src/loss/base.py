from typing import Callable, TypeVar, Union
from ..data import TrainingData
from torch import Tensor
from dataclasses import dataclass

FunctionName = TypeVar("FunctionName", bound=str)

@dataclass
class LossFunction:
    weights: list[Union[float, Callable[[float], float]]]
    losses: list[Callable]

    def __call__(self, data: TrainingData, time: float = 0.0) -> dict[FunctionName, Tensor]:
        return {
            loss.__name__: (w(time) if callable(w) else w) * loss(data)
            for loss, w in zip(self.losses, self.weights)
        }
