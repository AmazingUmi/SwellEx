"""Explicit method-owned dependencies used by the shared neural runtime."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class MethodBindings:
    method: str
    model: ModuleType
    data: ModuleType
    paths: ModuleType
    dataset_class: type
    input_count_field: str

    def check_config_shape(self, config: Any, input_shape: tuple[int, int, int]) -> None:
        if (getattr(config, self.input_count_field), config.input_freq_bins) != input_shape[1:]:
            raise ValueError(f"Checkpoint input shape does not match current {self.method} data: {input_shape}.")
