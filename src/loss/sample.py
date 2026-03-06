import torch
from torch import Tensor
from scipy.spatial import cKDTree


def sample_volume(n: int, bounds: float, device: str = "cuda") -> Tensor:
    return torch.rand(n, 3, device=device) * bounds * 2 - bounds


def sample_near_surface(surface_points: Tensor) -> Tensor:
    points_np = surface_points.detach().cpu().numpy()
    tree = cKDTree(points_np)

    k_neighbors = min(51, len(points_np))
    dist, _ = tree.query(points_np, k=k_neighbors, workers=-1)

    sigmas = dist[:, -1:]
    sigmas_tensor = torch.tensor(sigmas, device=surface_points.device, dtype=surface_points.dtype)

    noise = torch.randn_like(surface_points)
    near_points = surface_points + sigmas_tensor * noise
    return near_points
