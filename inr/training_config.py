import torch
from dataclasses import field
from pydantic import BaseModel
from typing import Callable
from torch import Tensor, nn

from .flexible_loss_weight import FlexibleLossWeight

LossFunction = Callable[[nn.Module, "TrainingConfig", Tensor, float], dict[str, Tensor]]


class LossWeights(BaseModel):
    dirichlet: FlexibleLossWeight = FlexibleLossWeight(7000.0)
    eikonal: FlexibleLossWeight = FlexibleLossWeight(50.0)
    developable: FlexibleLossWeight = FlexibleLossWeight(10.0)
    dnm: FlexibleLossWeight = FlexibleLossWeight(600.0)
    ncr: FlexibleLossWeight = FlexibleLossWeight(10.0)
    nsh: FlexibleLossWeight = FlexibleLossWeight(10.0)


class TrainingConfig(BaseModel):
    loss_function: Callable

    mesh_input_path: str = "data/abc/00800003.obj"

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

    testing: bool = False
    reconstruction_resolution: int = 256
    visualize: bool = False
    output_path: str | None = None
