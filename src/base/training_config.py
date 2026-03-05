import torch
from dataclasses import dataclass, field, fields
from typing import Callable

FlexibleLossWeight = Callable[[float], float] | float


def ncr_linear_decay(t: float) -> float:
    if t < 0.2:
        return 10.0
    if t < 0.5:
        return 10.0 + (0.001 - 10.0) * (t - 0.2) / (0.5 - 0.2)
    if t < 1.0:
        return 0.001 + (0.0 - 0.001) * (t - 0.5) / (1.0 - 0.5)
    return 0.0


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
    mesh_input_path: str
    loss_function: Callable

    hidden_dim: int = 256
    hidden_layers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 10000

    loss_weights: LossWeights = field(default_factory=LossWeights)

    dnm_alpha: float = 100.0

    bidirectional_ncr: bool = True

    surface_points: int = 20000
    near_surface_points: int = 10000
    volume_points: int = 10000

    learning_rate: float = 5e-5

    volume_bounds: float = 1.1

    reconstruction_resolution: int = 256
    visualize: bool = False
    output_path: str | None = None
