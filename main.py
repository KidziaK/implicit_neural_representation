import torch
import open3d as o3d
from torch import optim
from reconstruction import extract_and_visualize_mesh
from src.base.training_config import TrainingConfig
from src.sdf_net import SDFNet, ActivationType
from src.training import train
import src.loss as loss
from src.loss.developable import developable
from src.io.load import load_point_cloud_from_mesh_file
from loguru import logger

def run_experiment(config: TrainingConfig):
    model = SDFNet(
        in_features=3,
        hidden_dim=256,
        hidden_layers=4,
        activation_type=ActivationType.SIREN
    ).to(config.device)

    optimizer = optim.Adam(model.parameters())

    surface_points = load_point_cloud_from_mesh_file(
        mesh_file_path=config.mesh_input_path,
        n=config.surface_points,
        device=config.device
    )

    result = train(
        model=model,
        config=config,
        loss_function=config.loss_function,
        optimizer=optimizer,
        surface_points=surface_points
    )

    logger.info("Training Done", f"Total training time: {result.training_time_s}s")

    mesh = extract_and_visualize_mesh(
        model=model,
        config=config,
    )

    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training on CPU")

    training_config = TrainingConfig(
        mesh_input_path=r"C:\Users\kidzi\Downloads\abc_0000_obj_v00\00000008\00000008_9b3d6a97e8de4aa193b81000_trimesh_000.obj",
        epochs=1000,
        loss_function=loss.developable
    )

    run_experiment(training_config)
