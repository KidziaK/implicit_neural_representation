import operator
from abc import ABC, abstractmethod
from typing import Callable, Any, Union
from ..data import TrainingData
from torch import Tensor

class LossFunction(ABC):
    @abstractmethod
    def __call__(self, data: TrainingData) -> Tensor: ...

    def __add__(self, other: Union["LossFunction", float]) -> "LossFunction":
        return _BinaryOpLoss(self, other, operator.add)

    def __radd__(self, other: float) -> "LossFunction":
        return _BinaryOpLoss(other, self, operator.add)

    def __sub__(self, other: Union["LossFunction", float]) -> "LossFunction":
        return _BinaryOpLoss(self, other, operator.sub)

    def __rsub__(self, other: float) -> "LossFunction":
        return _BinaryOpLoss(other, self, operator.sub)

    def __mul__(self, other: Union["LossFunction", float]) -> "LossFunction":
        return _BinaryOpLoss(self, other, operator.mul)

    def __rmul__(self, other: float) -> "LossFunction":
        return _BinaryOpLoss(other, self, operator.mul)

    def __truediv__(self, other: Union["LossFunction", float]) -> "LossFunction":
        return _BinaryOpLoss(self, other, operator.truediv)

    def __rtruediv__(self, other: float) -> "LossFunction":
        return _BinaryOpLoss(other, self, operator.truediv)

class _BinaryOpLoss(LossFunction):
    def __init__(
        self, 
        left: Union["LossFunction", float], 
        right: Union["LossFunction", float], 
        op: Callable[[Any, Any], Tensor]
    ):
        self.left = left
        self.right = right
        self.op = op

    def __call__(self, data: TrainingData) -> Tensor:
        left_val = self.left if isinstance(self.left, (float, int)) else self.left(data)
        right_val = self.right if isinstance(self.right, (float, int)) else self.right(data)
        return self.op(left_val, right_val)
