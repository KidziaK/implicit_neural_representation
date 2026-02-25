import torch

from . import SimpleTrainingData
from .base import DataSampler

class CircleSampler(DataSampler):
    # TODO refactor, lot of repeated code
    def __init__(
        self,
        sampled_surface_points_num: int = 30000,
        on_manifold_points_num: int = 10000,
        off_manifold_points_num: int = 10000
    ) -> None:
        random_normal = torch.randn(sampled_surface_points_num, 2)
        self.sampled_surface_points = random_normal / random_normal.norm(dim=1, keepdim=True)

        self.on_manifold_points_num = on_manifold_points_num
        self.off_manifold_points_num = off_manifold_points_num

    def sample(self) -> SimpleTrainingData:
        n = self.sampled_surface_points.shape[0]
        index = torch.randperm(n)[:self.on_manifold_points_num]
        on_manifold_points = self.sampled_surface_points[index]
        off_manifold_points = torch.rand(self.off_manifold_points_num, 2) * 2 - 1
        return SimpleTrainingData(on_manifold_points, off_manifold_points)
