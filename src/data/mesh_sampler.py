import numpy as np
import torch
from .base import DataSampler
from .simple_training_data import SimpleTrainingData
from pathlib import Path
import open3d as o3d

class MeshSampler(DataSampler):
    def __init__(
        self,
        mesh_path: Path | str,
        sampled_surface_points_num: int = 30000,
        on_manifold_points_num: int = 10000,
        off_manifold_points_num: int = 10000
    ) -> None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = mesh.sample_points_uniformly(sampled_surface_points_num)
        points_numpy = np.asarray(pcd.points)
        points_tensor = torch.from_numpy(points_numpy).float()
        self.sampled_surface_points = points_tensor

        self.on_manifold_points_num = on_manifold_points_num
        self.off_manifold_points_num = off_manifold_points_num

    def sample(self) -> SimpleTrainingData:
        n = self.sampled_surface_points.shape[0]
        index = torch.randperm(n)[:self.on_manifold_points_num]
        on_manifold_points = self.sampled_surface_points[index]
        off_manifold_points = torch.rand(self.off_manifold_points_num, 3) * 2 - 1
        return SimpleTrainingData(on_manifold_points, off_manifold_points)
