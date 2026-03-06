import pytest
from inr.loss import losses
from inr.base.loss import LossFunction
from inr.base.training_config import TrainingConfig
from inr.experiment import run_experiment


@pytest.mark.parametrize("loss", losses)
def test_custom_loss(loss: LossFunction):
    print(TrainingConfig.testing_mode)
    run_experiment(TrainingConfig(loss_function=loss, epochs=1))
