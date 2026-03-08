import torch
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Callable, Union
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class FlexibleLossWeight:
    def __init__(self, value: Union[float, Callable[[float], float], str] = 0.0):
        self._source_value = value

        if isinstance(value, str):
            value = value.strip()
            # If it's a lambda string expression, eval works fine
            if value.startswith("lambda"):
                self.value = eval(value)
            else:
                # If it's a full function definition (def ...), we must exec it
                local_namespace = {}
                exec(value, {}, local_namespace)

                # Extract the first callable found in the execution namespace
                funcs = [v for v in local_namespace.values() if callable(v)]
                if funcs:
                    self.value = funcs[0]
                else:
                    raise ValueError("No callable function found in the provided source code.")
        else:
            self.value = value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self)

    def __set__(self, instance, value):
        if isinstance(value, FlexibleLossWeight):
            instance.__dict__[self.name] = value
        else:
            instance.__dict__[self.name] = FlexibleLossWeight(value)

    def __set_name__(self, owner, name):
        self.name = name

    def __call__(self, t: float) -> float:
        if callable(self.value):
            return float(self.value(t))
        return float(self.value)

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:

        def validate_flexible_weight(value: Any) -> "FlexibleLossWeight":
            if isinstance(value, cls):
                return value
            return cls(value)

        def serialize_flexible_weight(instance: "FlexibleLossWeight") -> Union[float, str]:
            # If initialized with a float, return the float
            if isinstance(instance.value, (float, int)):
                return float(instance.value)

            # If we already have the source string, use it
            if isinstance(instance._source_value, str):
                return instance._source_value

            # Introspect the callable to get its source code
            if callable(instance.value):
                try:
                    # Get the source code
                    source = inspect.getsource(instance.value)
                    # Dedent removes leading whitespace from nested functions
                    return textwrap.dedent(source).strip()
                except (OSError, TypeError) as e:
                    raise ValueError(
                        f"Failed to introspect source code for {instance.value}. "
                        f"Ensure the function is defined in a file, not dynamically in a REPL. Error: {e}"
                    )

            return float(instance.value)

        return core_schema.no_info_after_validator_function(
            validate_flexible_weight,
            core_schema.union_schema([
                core_schema.float_schema(),
                core_schema.str_schema(),
                core_schema.callable_schema(),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_flexible_weight,
                info_arg=False,
                return_schema=core_schema.union_schema([
                    core_schema.float_schema(),
                    core_schema.str_schema()
                ])
            )
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
