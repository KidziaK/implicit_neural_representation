import torch
from dataclasses import dataclass, field, fields
from typing import Callable

FlexibleLossWeight = Callable[[float], float] | float

class LambdaConverterMeta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            for f in fields(self):
                value = getattr(self, f.name)
                if isinstance(value, (float, int)):
                    def make_lambda(v):
                        return lambda x: v
                    setattr(self, f.name, make_lambda(value))

        cls.__init__ = __init__
        return cls

@dataclass
class LossWeights(metaclass=LambdaConverterMeta):
    dirichlet: FlexibleLossWeight = 7000.0
    eikonal: FlexibleLossWeight = 50.0
    developable: FlexibleLossWeight = 10.0
    dnm: FlexibleLossWeight = 600.0
    ncr: FlexibleLossWeight = 10.0
    nsh: FlexibleLossWeight = 10.0

@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 10000

    loss_weights: LossWeights = field(default_factory=LossWeights)

    dnm_alpha: float = 100.0

    surface_points: int = 20000
    manifold_points: int = 10000
    volume_points: int = 5000

    learning_rate: float = 5e-5

    volume_bounds: float = 1.1
