import torch

from . import SimpleTrainingData
from .base import DataSampler

class CircleSampler(DataSampler):
    def __init__(
        self,
        on_manifold_points_num: int = 4000,
        off_manifold_points_num: int = 3000
    ) -> None:
        random_normal = torch.randn(on_manifold_points_num, 2)
        self.on_manifold_points = .5 * random_normal / random_normal.norm(dim=1, keepdim=True)
        self.off_manifold_points = torch.rand(off_manifold_points_num, 2) * 2 - 1

    def sample(self) -> SimpleTrainingData:
        return SimpleTrainingData(self.on_manifold_points, self.off_manifold_points)
