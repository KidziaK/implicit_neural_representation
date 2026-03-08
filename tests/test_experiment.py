from inr.experiment import run_experiment
from inr.training_config import TrainingConfig, LossWeights
from pathlib import Path


def test_experiment() -> None:
    config = TrainingConfig(epochs=1, loss_weights=LossWeights(dirichlet=1), hidden_dim=8, hidden_layers=1)
    run_experiment(config, input_path=Path("data/abc/00000003.obj"), skip_reconstruction=True)
