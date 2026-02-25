import torch

from src.data import CircleSampler, SquareSampler
from src.experiment import Experiment
from src.loss.base import LossFunction
from src.loss import dirichlet_on_manifold_loss, dirichlet_off_manifold_loss, eikonal_loss_l1, eikonal_loss_l2, dirichlet_loss
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig
from src.visualize import plot_2d_sdf

if __name__ == "__main__":
    model = SDFNet(
        in_dims=2,
        hidden_dim=64,
        hidden_layers_num=4,
        out_dims=1
    )
    loss_function = LossFunction(
        weights=[1., .2],
        losses=[dirichlet_on_manifold_loss, eikonal_loss_l2]
    )
    data_sampler = SquareSampler(on_manifold=3000, off_manifold=4000)
    training_config = TrainingConfig(epochs=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = Experiment(model, loss_function, data_sampler, training_config, optimizer)
    experiment.train()

    res = 256
    a = 1.
    interval = torch.linspace(-a, a, res)
    grid = torch.stack(torch.meshgrid(interval, interval, indexing='ij'), dim=-1)
    sdf = experiment.evaluate(grid)
    plot_2d_sdf(sdf)
