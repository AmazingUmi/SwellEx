"""Method-owned dependencies used by the shared neural runtime."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

from scripts.data_processing.input_contract import METHODS

INPUT_COUNT_FIELD = {"RBD": "input_elements", "SCM": "input_pairs"}


@dataclass(frozen=True)
class MethodBindings:
    method: str
    model: ModuleType
    data: ModuleType
    dataset_class: type
    input_count_field: str

    def check_config_shape(self, config: Any, input_shape: tuple[int, int, int]) -> None:
        if (getattr(config, self.input_count_field), config.input_freq_bins) != input_shape[1:]:
            raise ValueError(f"Checkpoint input shape does not match current {self.method} data: {input_shape}.")


def bindings_for(method: str) -> MethodBindings:
    if method not in METHODS:
        raise ValueError(f"METHOD must be one of {', '.join(METHODS)}")
    from . import data, model
    return MethodBindings(method, model, data, data.DATASET_CLASSES[method], INPUT_COUNT_FIELD[method])
