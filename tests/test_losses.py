import pytest
from inr.loss import losses
from inr.training_config import TrainingConfig, LossFunction
from inr.experiment import run_experiment


@pytest.mark.parametrize("loss", losses)
def test_custom_loss(loss: LossFunction):
    config = TrainingConfig(
        loss_function=loss, epochs=1, hidden_dim=1, testing=True, device="cpu"
    )
    run_experiment(config)
