import torch
from dataclasses import dataclass, field
from torch import nn, optim, Tensor
from .loss.base import LossFunction
from .training_config import TrainingConfig, VisualizationConfig
from .data import DataSampler
from .logger import get_logger
from .model.projection import project_last_layer_to_zero_on_surface
from .visualize import show, extract_mesh
import os

logger = get_logger(__name__)

@dataclass
class Experiment:
    model: nn.Module
    loss_function: LossFunction
    data_sampler: DataSampler
    training_config: TrainingConfig
    optimizer: optim.Optimizer
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)

    def train(self) -> None:
        self.model.train()
        logger.info(f"Starting training with loss function: {self.loss_function}")

        for epoch in range(self.training_config.epochs):
            self.optimizer.zero_grad()

            data = self.data_sampler.sample()
            data.grad(self.model)

            t = epoch / max(1, self.training_config.epochs - 1)
            loss_dict = self.loss_function(data, time=t)
            loss = torch.stack(list(loss_dict.values())).sum(dim=0)

            y = "\033[33m"
            r = "\033[0m"

            loss_list = [f"{name}: {y}{tensor.item():.4f}{r}" for name, tensor in loss_dict.items()]
            lost_str = f"total: {y}{loss.item():.4f}{r}, " + ", ".join(loss_list)

            loss.backward()

            self.optimizer.step()

            proj_str = ""
            if self.training_config.use_projection and (epoch % max(1, self.training_config.proj_every) == 0):
                proj_metrics = project_last_layer_to_zero_on_surface(
                    self.model,
                    data.on_manifold_points,
                    eps=self.training_config.proj_eps,
                    cap_n=self.training_config.proj_cap_n
                )
                proj_str = f" | Proj max|f|: {proj_metrics['pre_max']:.2e} -> {proj_metrics['post_max']:.2e}"

            logger.info(f"Epoch {y}{epoch + 1}{r}/{self.training_config.epochs} - {lost_str}{proj_str}")

            if self.visualization_config.visualize and (epoch + 1) % self.visualization_config.every == 0:
                show(self, epoch=epoch + 1)
        
        # Save final mesh at the end of training if in 3D mode
        if self.model.in_dims == 3:
            import open3d as o3d
            try:
                mesh = extract_mesh(self)
                os.makedirs("visualization", exist_ok=True)
                final_path = "visualization/final_mesh.stl"
                o3d.io.write_triangle_mesh(final_path, mesh)
                logger.info(f"Saved final 3D reconstruction mesh to {final_path}")
            except Exception as e:
                logger.error(f"Failed to extract and save final mesh: {e}")

    def evaluate(self, x: Tensor) -> Tensor:
        self.model.eval()
        flat_x = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            flat_sdf = self.model(flat_x)
        return flat_sdf.reshape(x.shape[:-1])
