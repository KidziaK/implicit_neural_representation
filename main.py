import torch

from src.experiment import Experiment
from src.loss.base import LossFunction
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig
from src.visualize import show

import src.loss as loss
import src.data as data

if __name__ == "__main__":
    model = SDFNet(
        in_dims=2,
        hidden_dim=64,
        hidden_layers_num=4,
        out_dims=1
    )
    loss_function = LossFunction(
        weights=[1., .2],
        losses=[loss.dirichlet_on_manifold_loss, loss.eikonal_loss_l2]
    )
    data_sampler = data.CircleSampler(on_manifold=3000, off_manifold=4000)
    training_config = TrainingConfig(epochs=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = Experiment(model, loss_function, data_sampler, training_config, optimizer)
    experiment.train()

    show(experiment)
