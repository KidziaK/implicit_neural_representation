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

    def _to_json_compatible(self) -> float | dict[str, Any] | None:
        if self._is_constant:
            return self._value
        return None

    @classmethod
    def _from_json_compatible(cls, data: float | dict[str, Any] | None) -> "FlexibleLossWeight":
        if data is None:
            raise TypeError(
                "Cannot deserialize loss weight from JSON null. "
                "Use a constant number or a schedule dict (e.g. {\"type\": \"step\", \"before\": 0, \"after\": 10, \"t_threshold\": 0.5})."
            )
        if isinstance(data, (int, float)):
            return cls(float(data))
        if isinstance(data, dict):
            if data.get("type") == "callable":
                raise TypeError(
                    "Cannot deserialize callable loss weight from JSON. "
                    "Use a constant number or a schedule dict (e.g. {\"type\": \"step\", \"before\": 0, \"after\": 10, \"t_threshold\": 0.5})."
                )
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

        def serialize(value: FlexibleLossWeight) -> float | dict[str, Any] | None:
            return value._to_json_compatible()

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )
