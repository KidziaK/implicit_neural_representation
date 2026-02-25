import torch
from dataclasses import dataclass
from torch import Tensor, nn
from .base import TrainingData

@dataclass
class SimpleTrainingData(TrainingData):
    on_manifold_points: Tensor
    off_manifold_points: Tensor

    on_manifold_points_sdf: Tensor | None = None
    off_manifold_points_sdf: Tensor | None = None

    on_manifold_points_grad: Tensor | None = None
    off_manifold_points_grad: Tensor | None = None

    def sdf(self, model: nn.Module) -> None:
        all_points = torch.cat([self.on_manifold_points, self.off_manifold_points])
        sdf = model(all_points)
        self.on_manifold_points_sdf = sdf[:self.on_manifold_points.shape[0]]
        self.off_manifold_points_sdf = sdf[self.on_manifold_points.shape[0]:]

    def grad(self, model: nn.Module) -> None:
        all_points = torch.cat([self.on_manifold_points, self.off_manifold_points])
        sdf = model(all_points)
        grad = torch.autograd.grad(sdf, all_points, create_graph=True)[0]
        self.on_manifold_points_grad = grad[:self.on_manifold_points.shape[0]]
        self.off_manifold_points_grad = grad[self.on_manifold_points.shape[0]:]
