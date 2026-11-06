"""Model registry and construction helpers for both feature methods.

Registry names follow `scripts.network_training.catalog`: RBD is unprefixed
(`complex_cnn_range`) and SCM carries `scm_` (`scm_complex_cnn_range`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, fields
import importlib
from typing import TYPE_CHECKING, Any

from scripts.network_training.catalog import NEURAL_MODELS, neural_model_name
from .models.configs import RbdCnnConfig, RbdResNetConfig, ScmCnnConfig, ScmResNetConfig

if TYPE_CHECKING:
    from torch import nn

_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "complex_CNN": (".models.complex_cnn", "ComplexRangeCNN"),
    "ResNet18": (".models.resnet", "ResNet18Range"),
    "ResNet50": (".models.resnet", "ResNet50Range"),
}
_CONFIG_CLASSES: dict[str, dict[str, type]] = {
    "RBD": {"complex_CNN": RbdCnnConfig, "ResNet18": RbdResNetConfig, "ResNet50": RbdResNetConfig},
    "SCM": {"complex_CNN": ScmCnnConfig, "ResNet18": ScmResNetConfig, "ResNet50": ScmResNetConfig},
}


class _LazyModelRegistry(Mapping[str, tuple[type, type]]):
    """Mapping from registry model name to `(config_cls, model_cls)` loaded on demand."""

    def __init__(
        self,
        entries: dict[str, tuple[type, str]],
        specs: dict[str, tuple[str, str]],
    ) -> None:
        self._entries = dict(entries)
        self._specs = dict(specs)
        self._class_cache: dict[str, type] = {}

    def _resolve_model_class(self, architecture: str) -> type:
        cached = self._class_cache.get(architecture)
        if cached is not None:
            return cached
        module_name, class_name = self._specs[architecture]
        module = importlib.import_module(module_name, package=__package__)
        model_cls = getattr(module, class_name)
        self._class_cache[architecture] = model_cls
        return model_cls

    def config_class(self, model_name: str) -> type:
        config_cls, _ = self._entries[model_name]
        return config_cls

    def __getitem__(self, model_name: str) -> tuple[type, type]:
        config_cls, architecture = self._entries[model_name]
        return config_cls, self._resolve_model_class(architecture)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._entries)!r})"


MODEL_REGISTRY = _LazyModelRegistry(
    {
        neural_model_name(method, architecture): (configs[architecture], architecture)
        for method, configs in _CONFIG_CLASSES.items()
        for architecture in NEURAL_MODELS
    },
    _MODEL_SPECS,
)
MODEL_METHOD = {
    neural_model_name(method, architecture): method
    for method in _CONFIG_CLASSES for architecture in NEURAL_MODELS
}
MODEL_DATASET_INPUT_NORM = {name: True for name in MODEL_REGISTRY}


def models_for(method: str) -> list[str]:
    return sorted(name for name, owner in MODEL_METHOD.items() if owner == method)


def _lookup(model_name: str, what: str = "model") -> tuple[type, type]:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown {what}: {model_name}") from exc


def _lookup_config(model_name: str, what: str = "model") -> type:
    try:
        return MODEL_REGISTRY.config_class(model_name)
    except KeyError as exc:
        raise ValueError(f"Unknown {what}: {model_name}") from exc


def build_model_config(
    model_name: str,
    input_shape: tuple[int, int, int],
    **model_kwargs: Any,
) -> Any:
    config_cls = _lookup_config(model_name)
    _, rows, freq_bins = input_shape
    config_fields = {field.name for field in fields(config_cls)}
    count_field = "input_elements" if "input_elements" in config_fields else "input_pairs"
    config_values = {count_field: rows, "input_freq_bins": freq_bins}
    config_values.update(
        {
            key: value
            for key, value in model_kwargs.items()
            if key in config_fields and value is not None
        }
    )
    return config_cls(**config_values)


def build_model(model_name: str, config: Any) -> nn.Module:
    _, model_cls = _lookup(model_name)
    return model_cls(config)


def model_uses_dataset_input_norm(model_name: str) -> bool:
    try:
        return MODEL_DATASET_INPUT_NORM[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[str, Any]:
    model_name = str(checkpoint["model_name"])
    config_cls = _lookup_config(model_name, "checkpoint model")
    return model_name, config_cls(**checkpoint["model_config"])


def serialize_model_config(config: Any) -> dict[str, Any]:
    return asdict(config)
