import torch
from src.base.training_config import TrainingConfig, LossWeights
import src.loss as loss
from loguru import logger
from src.experiment import run_experiment
from pathlib import Path
from itertools import product


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training on CPU")

    experiment_name = "dummy"
    parts = ["00800003"]
    losses = [loss.digs_loss, loss.developable_loss]

    for loss_function, part_name in product(losses, parts):
        output_path = Path(f"output/abc/{experiment_name}/{part_name}_{loss_function.__name__}.obj")
        if output_path.exists():
            continue

        training_config = TrainingConfig(
            mesh_input_path=Path(f"data/abc/{part_name}.obj"),
            epochs=100,
            loss_function=loss_function,
            volume_points=10000,
            loss_weights=LossWeights(eikonal=lambda t: t),
            output_path=output_path,
        )

        run_experiment(training_config)
