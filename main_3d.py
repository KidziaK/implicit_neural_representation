import torch

from src.data.mesh_sampler import MeshSampler
from src.experiment import Experiment
from src.loss import LossFunction
from src.loss import eikonal_loss_l2, dirichlet_on_manifold_loss, true_distance_loss

from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig, VisualizationConfig
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    model = SDFNet(in_dims=3)

    # Note: Bunny will automatically be centered and scaled into [-0.5, 0.5]^3 natively by our MeshSampler update
    data_sampler = MeshSampler(
        mesh_path="data/00800035_259f5391c3947700164e504e_trimesh_009.obj",
        sampled_surface_points_num=50000,
        on_manifold_points_num=10000,
        off_manifold_points_num=10000,
    )

    loss_function = LossFunction(
        weights=[
            10,
            lambda t: 10 * t,
            lambda t: (1 - t) * 10
        ],
        losses=[
            dirichlet_on_manifold_loss,
            eikonal_loss_l2,
            true_distance_loss
        ]
    )

    training_config = TrainingConfig(
        epochs=1000,
        use_projection=False,
        proj_every=10,
        proj_eps=1e-5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    experiment = Experiment(
        model=model,
        loss_function=loss_function,
        data_sampler=data_sampler,
        training_config=training_config,
        optimizer=optimizer,
        visualization_config=VisualizationConfig(visualize=False, every=250, resolution=256)
    )

    logger.info("Initializing 3D SDF Training for Stanford Bunny...")
    experiment.train()

if __name__ == "__main__":
    main()
