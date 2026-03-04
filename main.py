from torch import optim
from src.base.training_config import TrainingConfig
from src.sdf_net import SDFNet, ActivationType
from src.training import train
from src.loss.digs import digs
from src.io.load import load_point_cloud_from_mesh_file
from loguru import logger

if __name__ == "__main__":
    model = SDFNet(
        in_features=3,
        hidden_dim=256,
        hidden_layers=4,
        activation_type=ActivationType.SIREN
    )

    config = TrainingConfig(
        epochs=1000
    )

    optimizer = optim.Adam(model.parameters())

    surface_points = load_point_cloud_from_mesh_file(
        mesh_file_path=r"C:\Users\kidzi\Downloads\abc_0000_obj_v00\00000008\00000008_9b3d6a97e8de4aa193b81000_trimesh_000.obj",
        n=config.surface_points,
        device=config.device
    )

    result = train(
        model=model,
        config=config,
        loss_function=digs,
        optimizer=optimizer,
        surface_points=surface_points
    )

    logger.info("Training Done", training_time_s=result.training_time_s)
