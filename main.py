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

    seeds = [42]
    experiment_name = "dummy"
    epochs = 10000
    parts = ["00800003"]
    losses = [loss.developable_loss]

    for loss_function, part_name, seed in product(losses, parts, seeds):
        output_path = Path(f"output/abc/{experiment_name}/{part_name}_{loss_function.__name__}_{epochs}.obj")

        if output_path.exists():
            continue

        torch.manual_seed(seed)

        training_config = TrainingConfig(
            mesh_input_path=Path(f"data/abc/{part_name}.obj"),
            epochs=epochs,
            loss_function=loss_function,
            volume_points=10000,
            loss_weights=LossWeights(),
            output_path=output_path,
        )

        run_experiment(training_config)
