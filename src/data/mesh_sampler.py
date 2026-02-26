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
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        # Normalize mesh to fit perfectly inside [-0.5, 0.5]^3
        mesh.translate(-mesh.get_center())
        max_extent = max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.scale(1.0 / max_extent, center=(0, 0, 0))

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

        # 3D Domain boundary points explicitly sampled along faces of [-1, 1] cube
        domain_pts_per_face = max(1, self.on_manifold_points_num // 6)
        pos_d1 = torch.rand(domain_pts_per_face, 2) * 2 - 1
        domain_points_list = []
        for fixed_dim in range(3):
            for sign in [-1.0, 1.0]:
                pts = torch.zeros(domain_pts_per_face, 3)
                idx = 0
                for d in range(3):
                    if d == fixed_dim:
                        pts[:, d] = sign
                    else:
                        pts[:, d] = pos_d1[:, idx]
                        idx += 1
                domain_points_list.append(pts)
        domain_points = torch.cat(domain_points_list, dim=0)
        domain_idx = torch.randperm(domain_points.shape[0])
        domain_boundary_points = domain_points[domain_idx]

        dists = torch.cdist(domain_boundary_points, on_manifold_points)
        min_dists, _ = dists.min(dim=1, keepdim=True)

        return SimpleTrainingData(on_manifold_points, off_manifold_points, 
                                  domain_boundary_points=domain_boundary_points, 
                                  domain_boundary_distances=min_dists)
