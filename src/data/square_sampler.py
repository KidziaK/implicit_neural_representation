import torch

from . import SimpleTrainingData
from .base import DataSampler

class SquareSampler(DataSampler):
    def __init__(self, on_manifold: int, off_manifold: int) -> None:
        pts_per_side = on_manifold // 4
        pos = torch.linspace(-0.5, 0.5, pts_per_side)
        
        points_list = []
        for i in range(4):
            if i == 0: 
                pts = torch.stack([torch.full_like(pos, 0.5), pos], dim=1)
            elif i == 1: 
                pts = torch.stack([pos, torch.full_like(pos, 0.5)], dim=1)
            elif i == 2: 
                pts = torch.stack([torch.full_like(pos, -0.5), pos], dim=1)
            elif i == 3: 
                pts = torch.stack([pos, torch.full_like(pos, -0.5)], dim=1)
            points_list.append(pts)
            
        points = torch.cat(points_list, dim=0)
        dim = 0
        idx = torch.randperm(points.shape[dim])

        self.on_manifold_points = points[idx]
        self.off_manifold_points = torch.rand(off_manifold, 2) * 2 - 1

    def sample(self) -> SimpleTrainingData:
        return SimpleTrainingData(self.on_manifold_points, self.off_manifold_points)
