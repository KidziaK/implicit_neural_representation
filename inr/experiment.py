import json
import open3d as o3d
from torch import optim
from inr.reconstruction import extract_and_visualize_mesh
from inr.training_config import TrainingConfig
from inr.load import load_point_cloud_from_mesh_file
from inr.sdf_net import SDFNet, ActivationType
from inr.training import train
from loguru import logger
import numpy as np
from inr.measure import chamfer_distance, hausdorff_distance
from pathlib import Path


def run_experiment(config: TrainingConfig, visualize: bool = False) -> None:
    model = SDFNet(
        in_features=3,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        activation_type=ActivationType.SIREN,
    ).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    surface_points = load_point_cloud_from_mesh_file(
        mesh_file_path=config.mesh_input_path,
        n=config.surface_points,
        device=config.device,
    )

    result = train(
        model=model,
        config=config,
        loss_function=config.loss_function,
        optimizer=optimizer,
        surface_points=surface_points,
    )

    logger.info(f"Training Done, total training time: {int(result.training_time_s)}s")

    if config.testing:
        return

    mesh = extract_and_visualize_mesh(
        model=model,
        config=config,
    )

    logger.info("Sampling 100k points from original and reconstructed meshes for evaluation...")
    original_points_tensor = load_point_cloud_from_mesh_file(
        mesh_file_path=config.mesh_input_path,
        n=100000,
        bounds=config.volume_bounds,
        device="cpu",
    )
    original_points = original_points_tensor.numpy()

    reconstructed_pc = mesh.sample_points_uniformly(number_of_points=100000)
    reconstructed_points = np.asarray(reconstructed_pc.points)

    chamfer_dist = chamfer_distance(original_points, reconstructed_points)
    hausdorff_dist = hausdorff_distance(original_points, reconstructed_points)

    logger.info(f"Chamfer Distance: {chamfer_dist:.6f}")
    logger.info(f"Hausdorff Distance: {hausdorff_dist:.6f}")

    if visualize:
        o3d.visualization.draw_geometries([mesh])

    metadata = dict(
        training_time=result.training_time_s,
        chamfer_distance=chamfer_dist,
        hausdorff_distance=hausdorff_dist,
        config=config.model_dump(exclude={"loss_function"}),
    )

    json.dump(metadata, Path(config.output_path).with_suffix(".json").open(mode="w+"), indent=4)
