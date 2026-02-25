import torch

from src.data import MeshSampler
from src.experiment import Experiment
from src.loss.eikonal_loss import EikonalLoss
from src.model.sdf_net import SDFNet
from src.training_config import TrainingConfig

if __name__ == "__main__":
    model = SDFNet(out_dims=2)
    loss_function = EikonalLoss()
    data_sampler = MeshSampler(mesh_path="TODO")
    training_config = TrainingConfig(epochs=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    experiment = Experiment(model, loss_function, data_sampler, training_config, optimizer)
    experiment.train()
