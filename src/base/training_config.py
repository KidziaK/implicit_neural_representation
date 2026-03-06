import torch
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path


class FlexibleLossWeight:
    def __init__(self, value: float | Callable[[float], float] = 0.0):
        self.value = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self)

    def __set__(self, instance, value):
        if isinstance(value, FlexibleLossWeight):
            instance.__dict__[self.name] = value
        else:
            instance.__dict__[self.name] = FlexibleLossWeight(value)

    def __set_name__(self, owner, name):
        self.name = name

    def __call__(self, t: float) -> float:
        if callable(self.value):
            return float(self.value(t))
        return float(self.value)


@dataclass
class LossWeights:
    dirichlet: FlexibleLossWeight = FlexibleLossWeight(7000.0)
    eikonal: FlexibleLossWeight = FlexibleLossWeight(50.0)
    developable: FlexibleLossWeight = FlexibleLossWeight(10.0)
    dnm: FlexibleLossWeight = FlexibleLossWeight(600.0)
    ncr: FlexibleLossWeight = FlexibleLossWeight(10.0)
    nsh: FlexibleLossWeight = FlexibleLossWeight(10.0)


@dataclass
class TrainingConfig:
    mesh_input_path: str | Path
    loss_function: Callable

    hidden_dim: int = 256
    hidden_layers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 10000

    loss_weights: LossWeights = field(default_factory=LossWeights)

    dnm_alpha: float = 100.0

    bidirectional_ncr: bool = True

    surface_points: int = 20000
    volume_points: int = 10000

    learning_rate: float = 5e-5

    volume_bounds: float = 1.1

    reconstruction_resolution: int = 256
    visualize: bool = False
    output_path: str | Path | None = None
