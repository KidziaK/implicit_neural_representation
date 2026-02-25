import torch

from src.data import CircleSampler
from src.experiment import Experiment
from src.loss.eikonal_loss import EikonalLoss
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig

if __name__ == "__main__":
    model = SDFNet(in_dims=2)
    loss_function = EikonalLoss()
    data_sampler = CircleSampler()
    training_config = TrainingConfig(epochs=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = Experiment(model, loss_function, data_sampler, training_config, optimizer)
    experiment.train()

    res = 100
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='ij'), dim=-1)
    sdf = experiment.evaluate(grid)
