import torch
from inr.training_config import TrainingConfig, LossWeights
import inr.loss as loss
from loguru import logger
from inr.experiment import run_experiment
from pathlib import Path
from itertools import product


def ncr_linear_decay(t: float) -> float:
    if t < 0.2:
        return 10.0
    if t < 0.5:
        return 10.0 + (0.001 - 10.0) * (t - 0.2) / (0.5 - 0.2)
    if t < 1.0:
        return 0.001 + (0.0 - 0.001) * (t - 0.5) / (1.0 - 0.5)
    return 0.0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training on CPU")

    seeds = [42]
    experiment_name = "dummy"
    epochs = 100
    parts = ["00800003"]

    for part_name, seed in product(parts, seeds):
        output_path = Path(f"output/abc/{experiment_name}/{part_name}_{epochs}.glb")

        torch.manual_seed(seed)

        training_config = TrainingConfig(
            mesh_input_path=str(Path(f"data/abc/{part_name}.obj")),
            epochs=epochs,
            volume_points=10000,
            loss_weights=LossWeights(
                dirichlet=7000,
                dnm=600,
                eikonal=50,
                gaussian_curvature=ncr_linear_decay
            ),
            output_path=str(output_path),
        )

        run_experiment(training_config)
