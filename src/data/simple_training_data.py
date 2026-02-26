import torch
from dataclasses import dataclass
from torch import Tensor, nn
from .base import TrainingData

@dataclass
class SimpleTrainingData(TrainingData):
    on_manifold_points: Tensor
    off_manifold_points: Tensor
    domain_boundary_points: Tensor | None = None

    on_manifold_points_sdf: Tensor | None = None
    off_manifold_points_sdf: Tensor | None = None
    domain_boundary_points_sdf: Tensor | None = None

    on_manifold_points_grad: Tensor | None = None
    off_manifold_points_grad: Tensor | None = None

    domain_boundary_distances: Tensor | None = None

    def grad(self, model: nn.Module) -> None:
        tensors_to_cat = [self.on_manifold_points, self.off_manifold_points]
        if self.domain_boundary_points is not None:
             tensors_to_cat.append(self.domain_boundary_points)
            
        all_points = torch.cat(tensors_to_cat, dim=0)
        all_points = all_points.detach().requires_grad_(True)

        sdf = model(all_points)

        sdf_grad = torch.autograd.grad(
            outputs=sdf,
            inputs=all_points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        num_on = self.on_manifold_points.shape[0]
        num_off = self.off_manifold_points.shape[0]

        self.on_manifold_points_sdf = sdf[:num_on]
        self.off_manifold_points_sdf = sdf[num_on:num_on+num_off]

        self.on_manifold_points_grad = sdf_grad[:num_on]
        self.off_manifold_points_grad = sdf_grad[num_on:num_on+num_off]

        if self.domain_boundary_points is not None:
             self.domain_boundary_points_sdf = sdf[num_on+num_off:]