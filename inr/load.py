import torch
from torch import Tensor
import numpy as np
import open3d as o3d
from pathlib import Path


def load_point_cloud_from_mesh_file(
    mesh_file_path: str | Path,
    n: int = 20000,
    bounds: float = 1.0,
    device: str = "cuda",
) -> Tensor:
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    pc = mesh.sample_points_uniformly(number_of_points=n)
    points_np = np.asarray(pc.points)

    center = np.mean(points_np, axis=0)
    points_np -= center
    max_dist = np.max(np.linalg.norm(points_np, axis=1))
    points = bounds * (points_np / max_dist)

    return torch.tensor(points, dtype=torch.float32, device=device)
