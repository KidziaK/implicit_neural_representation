import torch
from dataclasses import dataclass, field, fields
from typing import Callable, ClassVar

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
                    setattr(self, f.name, _const_weight(float(value)))

        cls.__init__ = __init__
        return cls


def _const_weight(v: float) -> Callable[[float], float]:
    return lambda x: v


@dataclass
class LossWeights(metaclass=LambdaConverterMeta):
    dirichlet: FlexibleLossWeight = 7000.0
    eikonal: FlexibleLossWeight = 50.0
    developable: FlexibleLossWeight = 10.0
    dnm: FlexibleLossWeight = 600.0
    ncr: FlexibleLossWeight = 10.0
    nsh: FlexibleLossWeight = 10.0

    _weight_names: ClassVar[frozenset[str]] = frozenset({"dirichlet", "eikonal", "developable", "dnm", "ncr", "nsh"})

    def __setattr__(self, name: str, value: FlexibleLossWeight) -> None:
        if name in type(self)._weight_names and isinstance(value, (int, float)):
            value = _const_weight(float(value))
        super().__setattr__(name, value)


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
