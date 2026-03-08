import torch
from torch import Tensor


def sample_volume(n: int, bounds: float, device: str = "cuda") -> Tensor:
    return torch.rand(n, 3, device=device) * bounds * 2 - bounds


def compute_sigmas(surface_points: Tensor, k: int) -> Tensor:
    with torch.no_grad():
        dist_matrix = torch.cdist(surface_points, surface_points)
        sigmas, _ = torch.kthvalue(dist_matrix, k=k, dim=1, keepdim=True)
        return sigmas


def sample_near_surface(surface_points: Tensor, sigmas: Tensor) -> Tensor:
    with torch.no_grad():
        noise = torch.randn_like(surface_points)
        near_points = surface_points + sigmas * noise
        return near_points
