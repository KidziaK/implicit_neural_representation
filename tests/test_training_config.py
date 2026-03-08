import json
import pytest
from inr.base.training_config import FlexibleLossWeight, LossWeights, TrainingConfig


def test_flexible_loss_weight_constant_float():
    w = FlexibleLossWeight(7000.0)
    assert w(0.0) == 7000.0
    assert w(0.5) == 7000.0
    assert w(1.0) == 7000.0


def test_flexible_loss_weight_constant_int():
    w = FlexibleLossWeight(10)
    assert w(0.0) == 10.0


def test_flexible_loss_weight_callable():
    w = FlexibleLossWeight(lambda t: 0 if t < 0.5 else 10)
    assert w(0.0) == 0
    assert w(0.4) == 0
    assert w(0.5) == 10
    assert w(1.0) == 10


def test_flexible_loss_weight_dict_constant():
    w = FlexibleLossWeight({"type": "constant", "value": 42.0})
    assert w(0.0) == 42.0
    assert w(0.7) == 42.0


def test_flexible_loss_weight_dict_step():
    w = FlexibleLossWeight({"type": "step", "before": 0, "after": 10, "t_threshold": 0.5})
    assert w(0.0) == 0
    assert w(0.4) == 0
    assert w(0.5) == 10
    assert w(1.0) == 10


def test_flexible_loss_weight_dict_linear():
    w = FlexibleLossWeight({"type": "linear", "start": 0.0, "end": 1.0})
    assert w(0.0) == 0.0
    assert w(0.5) == 0.5
    assert w(1.0) == 1.0


def test_flexible_loss_weight_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown schedule type"):
        FlexibleLossWeight({"type": "unknown"})


def test_flexible_loss_weight_to_json_constant():
    w = FlexibleLossWeight(7000.0)
    assert w._to_json_compatible() == 7000.0


def test_flexible_loss_weight_to_json_callable():
    w = FlexibleLossWeight(lambda t: t)
    assert w._to_json_compatible() == {"type": "callable", "fallback": 1.0}


def test_flexible_loss_weight_from_json_float():
    w = FlexibleLossWeight._from_json_compatible(7000.0)
    assert w(0.0) == 7000.0


def test_flexible_loss_weight_from_json_callable_placeholder():
    w = FlexibleLossWeight._from_json_compatible({"type": "callable", "fallback": 5.0})
    assert w(0.0) == 5.0


def test_flexible_loss_weight_from_json_dict_schedule():
    w = FlexibleLossWeight._from_json_compatible({"type": "step", "before": 0, "after": 10, "t_threshold": 0.5})
    assert w(0.0) == 0
    assert w(0.6) == 10


def test_loss_weights_defaults():
    lw = LossWeights()
    assert lw.dirichlet(0.0) == 7000.0
    assert lw.eikonal(0.0) == 50.0
    assert lw.developable(0.0) == 10.0


def test_loss_weights_partial_override_callable():
    lw = LossWeights(developable=lambda t: 0 if t < 0.5 else 10)
    assert lw.developable(0.0) == 0
    assert lw.developable(0.6) == 10
    assert lw.dirichlet(0.0) == 7000.0


def test_loss_weights_partial_override_float():
    lw = LossWeights(developable=99.0)
    assert lw.developable(0.0) == 99.0


def test_loss_weights_model_dump_json_roundtrip_constants():
    lw = LossWeights()
    dumped = lw.model_dump(mode="json")
    assert isinstance(dumped["dirichlet"], (int, float))
    assert dumped["dirichlet"] == 7000.0
    j = lw.model_dump_json()
    parsed = json.loads(j)
    assert parsed["dirichlet"] == 7000.0


def test_training_config_model_dump_json_with_loss_weights():
    cfg = TrainingConfig(loss_function=lambda x: x, loss_weights=LossWeights())
    j = cfg.model_dump_json(exclude={"loss_function"})
    data = json.loads(j)
    assert "loss_weights" in data
    assert data["loss_weights"]["dirichlet"] == 7000.0


def test_training_config_model_dump_json_with_callable_weight():
    cfg = TrainingConfig(
        loss_function=lambda x: x,
        loss_weights=LossWeights(developable=lambda t: 10),
    )
    j = cfg.model_dump_json(exclude={"loss_function"})
    data = json.loads(j)
    assert data["loss_weights"]["developable"] == {"type": "callable", "fallback": 1.0}
