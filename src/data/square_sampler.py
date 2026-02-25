import torch

from . import SimpleTrainingData
from .base import DataSampler

class SquareSampler(DataSampler):
    def __init__(self, on_manifold: int, off_manifold: int) -> None:
        side_indices = torch.randint(0, 4, (on_manifold,))
        pos = torch.rand(on_manifold) - 0.5
        points = torch.zeros((on_manifold, 2))

        for i in range(4):
            mask = (side_indices == i)
            if i == 0: points[mask] = torch.stack(
                [torch.full_like(pos[mask], 0.5), pos[mask]], dim=1)
            if i == 1: points[mask] = torch.stack(
                [pos[mask], torch.full_like(pos[mask], 0.5)], dim=1)
            if i == 2: points[mask] = torch.stack(
                [torch.full_like(pos[mask], -0.5), pos[mask]], dim=1)
            if i == 3: points[mask] = torch.stack(
                [pos[mask], torch.full_like(pos[mask], -0.5)], dim=1)

        self.on_manifold_points = points
        self.off_manifold_points = torch.rand(off_manifold, 2) * 2 - 1


    def sample(self) -> SimpleTrainingData:
        return SimpleTrainingData(self.on_manifold_points, self.off_manifold_points)
