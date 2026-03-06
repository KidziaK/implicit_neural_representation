import torch
import open3d as o3d
from torch import optim
from src.base.training_config import TrainingConfig
import src.loss as loss
from loguru import logger
import numpy as np
from src.experiment import run_experiment


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training on CPU")

    training_config = TrainingConfig(
        mesh_input_path=r"/Users/mikolajkida/Documents/github/implicit_neural_representation/data/00808652_a5311ecc077bfdd2cc3d9aa7_trimesh_000.obj",
        epochs=100,
        loss_function=loss.developable,
        volume_points=10000,
    )

    run_experiment(training_config)
