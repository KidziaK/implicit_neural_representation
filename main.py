import torch
from inr.training_config import TrainingConfig, LossWeights
from loguru import logger
from inr.experiment import run_experiment
from pathlib import Path
from itertools import product


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training on CPU")
    else:
        torch.cuda.init()

    seeds = [42]
    experiment_name = "bonnet2"
    parts = ["00000003"]

    for part_name, seed in product(parts, seeds):
        input_path = Path(f"data/abc/{part_name}.obj")
        output_path = Path(f"output/abc/{experiment_name}/{part_name}.glb")

        torch.manual_seed(seed)

        training_config = TrainingConfig(
            epochs=10000,
            surface_points=15000,
            volume_points=10000,
            loss_weights=LossWeights(
                dirichlet=7000,
                dnm=600,
                eikonal=50,
                gauss_bonnet=lambda t: 5 * (1 - t)
            ),
        )

        run_experiment(training_config, input_path=input_path, output_path=output_path)
