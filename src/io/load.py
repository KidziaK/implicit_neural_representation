import torch
from torch import Tensor
import numpy as np
import open3d as o3d
from pathlib import Path

def load_point_cloud_from_mesh_file(mesh_file_path: str | Path, n: int = 20000, device: str = "cuda") -> Tensor:
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    pc = mesh.sample_points_uniformly(number_of_points=n)
    points_np = np.array(pc.points)
    return torch.from_numpy(points_np).float().to(device)
