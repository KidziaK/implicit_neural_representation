import torch
from dataclasses import field
from pydantic import BaseModel
from typing import Callable
from torch import Tensor, nn

from .flexible_loss_weight import FlexibleLossWeight

LossFunction = Callable[[nn.Module, "TrainingConfig", Tensor, float, Tensor], dict[str, Tensor]]


class LossWeights(BaseModel):
    dirichlet: FlexibleLossWeight | None = None
    eikonal: FlexibleLossWeight | None = None
    gauss_bonnet: FlexibleLossWeight | None = None
    dnm: FlexibleLossWeight | None = None
    gaussian_curvature: FlexibleLossWeight | None = None
    singular_hessian: FlexibleLossWeight | None = None


class TrainingConfig(BaseModel):
    mesh_input_path: str = "data/abc/00800003.obj"

    hidden_dim: int = 256
    hidden_layers: int = 4

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
