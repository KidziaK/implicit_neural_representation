import torch
from torch import Tensor
from scipy.spatial import cKDTree


def sample_volume(n: int, bounds: float, device: str = "cuda") -> Tensor:
    return torch.rand(n, 3, device=device) * bounds * 2 - bounds


def sample_near_surface(surface_points: Tensor) -> Tensor:
    num_points = surface_points.shape[0]
    k_neighbors = min(51, num_points)

    dist_matrix = torch.cdist(surface_points, surface_points)
    sigmas, _ = torch.kthvalue(dist_matrix, k=k_neighbors, dim=1, keepdim=True)

    noise = torch.randn_like(surface_points)
    near_points = surface_points + sigmas * noise

    return near_points
