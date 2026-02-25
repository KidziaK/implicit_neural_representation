import torch

from src.data import CircleSampler
from src.experiment import Experiment
from src.loss.base import LossFunction
from src.loss import eikonal_loss_l1, eikonal_loss_l2, dirichlet_loss
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig
from src.visualize import plot_2d_sdf

if __name__ == "__main__":
    torch.random.manual_seed(100)

    model = SDFNet(in_dims=2)
    loss_function = LossFunction(
        weights=[0.2, 1.0],
        losses=[eikonal_loss_l2, dirichlet_loss]
    )
    data_sampler = CircleSampler()
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
