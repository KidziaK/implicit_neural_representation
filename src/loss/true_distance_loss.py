import torch
from torch import Tensor
import torch.nn.functional as F
from ..data import TrainingData

def true_distance_loss(data: TrainingData) -> Tensor:
    sdf = data.domain_boundary_points_sdf
    true_dists = data.domain_boundary_distances

    # The absolute value of the predicted SDF should match the true distance to the shape boundary
    # predicted_magnitude = torch.abs(sdf)
    return F.mse_loss(sdf, true_dists)
