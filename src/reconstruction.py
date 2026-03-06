import torch
import numpy as np
import open3d as o3d
from .base.training_config import TrainingConfig
from torch import nn
from pathlib import Path
from skimage import measure
from loguru import logger


def extract_and_visualize_mesh(
    model: nn.Module,
    config: TrainingConfig,
) -> o3d.geometry.TriangleMesh:
    model.eval()

    bounds = config.volume_bounds
    resolution = config.reconstruction_resolution

    x_lin = np.linspace(-bounds, bounds, resolution)
    y_lin = np.linspace(-bounds, bounds, resolution)
    z_lin = np.linspace(-bounds, bounds, resolution)
    X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")

    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=config.device)

    sdf_values = []
    chunk_size = 100000

    logger.info("Evaluating SDF")

    with torch.no_grad():
        for i in range(0, grid_points_tensor.shape[0], chunk_size):
            chunk = grid_points_tensor[i : i + chunk_size]
            sdf_chunk = model(chunk)
            sdf_values.append(sdf_chunk.cpu().numpy())

    sdf_volume = np.concatenate(sdf_values).reshape((resolution, resolution, resolution))

    logger.info("Starting reconstruction of mesh")

    vertices, triangles, normals, values = measure.marching_cubes(sdf_volume, level=0.0)

    logger.info("Reconstruction completed")

    vertices = (vertices / (resolution - 1)) * (2 * bounds) - bounds

    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    reconstructed_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    reconstructed_mesh.compute_vertex_normals()

    output_path = config.output_path

    if output_path:
        logger.info(f"Saving reconstructed mesh to: {output_path}")
        o3d.io.write_triangle_mesh(Path(output_path), reconstructed_mesh)

    return reconstructed_mesh
