import torch
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Callable, Union
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


WeightInput = Union[float, Callable[[float], float], dict[str, Any]]


def _schedule_from_dict(data: dict[str, Any]) -> Callable[[float], float]:
    kind = data.get("type", "constant")
    if kind == "constant":
        return lambda t: float(data["value"])
    if kind == "step":
        before = float(data["before"])
        after = float(data["after"])
        t_threshold = float(data["t_threshold"])
        return lambda t: before if t < t_threshold else after
    if kind == "linear":
        start = float(data["start"])
        end = float(data["end"])
        return lambda t: start + (end - start) * t
    raise ValueError(f"Unknown schedule type: {kind!r}")


def _dict_from_callable(fn: Callable[[float], float]) -> dict[str, Any]:
    return {"type": "callable", "fallback": 1.0}


class FlexibleLossWeight:

    __slots__ = ("_value", "_is_constant")

    def __init__(self, value: WeightInput):
        if callable(value) and not isinstance(value, type):
            self._value = value
            self._is_constant = False
        elif isinstance(value, dict):
            self._value = _schedule_from_dict(value)
            self._is_constant = False
        else:
            self._value = float(value)
            self._is_constant = True

    def __call__(self, t: float) -> float:
        if self._is_constant:
            return self._value
        return self._value(t)

    def _to_json_compatible(self) -> float | dict[str, Any]:
        if self._is_constant:
            return self._value
        return _dict_from_callable(self._value)

    @classmethod
    def _from_json_compatible(cls, data: float | dict[str, Any]) -> "FlexibleLossWeight":
        if isinstance(data, (int, float)):
            return cls(float(data))
        if isinstance(data, dict):
            if data.get("type") == "callable":
                return cls(float(data.get("fallback", 1.0)))
            return cls(data)
        raise TypeError(f"Cannot build FlexibleLossWeight from {type(data)}")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate(value: Any) -> FlexibleLossWeight:
            if isinstance(value, FlexibleLossWeight):
                return value
            if isinstance(value, (int, float)):
                return cls(float(value))
            if callable(value) and not isinstance(value, type):
                return cls(value)
            if isinstance(value, dict):
                return cls(value)
            raise ValueError(
                "FlexibleLossWeight must be a float, a callable (float -> float), or a schedule dict"
            )

        def serialize(value: FlexibleLossWeight) -> float | dict[str, Any]:
            return value._to_json_compatible()

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )


class LossWeights(BaseModel):
    dirichlet: FlexibleLossWeight = FlexibleLossWeight(7000.0)
    eikonal: FlexibleLossWeight = FlexibleLossWeight(50.0)
    developable: FlexibleLossWeight = FlexibleLossWeight(10.0)
    dnm: FlexibleLossWeight = FlexibleLossWeight(600.0)
    ncr: FlexibleLossWeight = FlexibleLossWeight(10.0)
    nsh: FlexibleLossWeight = FlexibleLossWeight(10.0)


class TrainingConfig(BaseModel):
    loss_function: Callable

    mesh_input_path: str | Path = Path("data/abc/00800003.obj")

    hidden_dim: int = 256
    hidden_layers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 10000

    loss_weights: LossWeights = field(default_factory=LossWeights)

    dnm_alpha: float = 100.0

    bidirectional_ncr: bool = True

    surface_points: int = 20000
    volume_points: int = 10000

    learning_rate: float = 5e-5

    volume_bounds: float = 1.1

    testing: bool = False
    reconstruction_resolution: int = 256
    visualize: bool = False
    output_path: str | Path | None = None
