from dataclasses import field
from pydantic import BaseModel
from typing import Callable


def eval_weight(weight: float | Callable, t: float) -> float:
    match weight:
        case float():
            return weight
        case _:
            return weight(t)


class LossWeights(BaseModel):
    dirichlet: float | Callable | None = None
    eikonal: float | Callable | None = None
    gauss_bonnet: float | Callable | None = None
    dnm: float | Callable | None = None
    ncadr: float | Callable | None = None
    flatcad: float | Callable | None = None


class TrainingConfig(BaseModel):
    hidden_dim: int = 256
    hidden_layers: int = 4

    epochs: int = 10000
    learning_rate: float = 5e-5
    grad_clip: float = 10.0

    loss_weights: LossWeights = field(default_factory=LossWeights)

    dnm_alpha: float = 100.0

    surface_points: int = 20000
    volume_points: int = 10000

    volume_bounds: float = 1.1
    reconstruction_resolution: int = 256
