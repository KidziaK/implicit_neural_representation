import torch
from torch import Tensor

def sample_surface(surface_points: Tensor, size: int) -> Tensor:
    idx = torch.randperm(surface_points.shape[0])[:size]
    return surface_points[idx]

def sample_volume(n: int, bounds: float, device: str = "cuda") -> Tensor:
    return torch.rand(n, 3, device=device) * bounds * 2 - bounds
