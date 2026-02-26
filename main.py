import torch

from src.experiment import Experiment
from src.loss.base import LossFunction
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig
from src.visualize import show

import src.loss as loss
import src.data as data

if __name__ == "__main__":
    for _ in range(1):
        model = SDFNet(
            in_dims=2,
            hidden_dim=128,
            hidden_layers_num=4,
            out_dims=1
        )

        schedule = lambda t: 0 if t < 0.3 else 1

        loss_function = LossFunction(
            weights=[
                10,
                lambda t: 10 * t,
                0,
                lambda t: (1 - t) * 10
            ],
            losses=[
                loss.dirichlet_on_manifold_loss,
                loss.eikonal_loss_l2,
                loss.dirichlet_off_manifold_loss,
                loss.true_distance_loss
            ]
        )
        data_sampler = data.SquareSampler(on_manifold=3000, off_manifold=4000)
        training_config = TrainingConfig(epochs=1000, use_projection=False, proj_every=50, proj_eps=1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        experiment = Experiment(model, loss_function, data_sampler, training_config, optimizer)
        experiment.train()

