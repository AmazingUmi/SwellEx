"""Model registry and construction helpers for both feature methods.

Registry names follow `scripts.network_training.catalog`: RBD is unprefixed
(`complex_cnn_range`) and SCM carries `scm_` (`scm_complex_cnn_range`).
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from torch import nn

from scripts.network_training.catalog import NEURAL_MODELS, neural_model_name
from .models.complex_cnn import ComplexRangeCNN
from .models.configs import RbdCnnConfig, RbdResNetConfig, ScmCnnConfig, ScmResNetConfig
from .models.resnet import ResNet18Range, ResNet50Range

_MODEL_CLASSES = {"complex_CNN": ComplexRangeCNN, "ResNet18": ResNet18Range, "ResNet50": ResNet50Range}
_CONFIG_CLASSES = {
    "RBD": {"complex_CNN": RbdCnnConfig, "ResNet18": RbdResNetConfig, "ResNet50": RbdResNetConfig},
    "SCM": {"complex_CNN": ScmCnnConfig, "ResNet18": ScmResNetConfig, "ResNet50": ScmResNetConfig},
}

MODEL_REGISTRY = {
    neural_model_name(method, architecture): (configs[architecture], _MODEL_CLASSES[architecture])
    for method, configs in _CONFIG_CLASSES.items()
    for architecture in NEURAL_MODELS
}
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


def build_model_config(
    model_name: str,
    input_shape: tuple[int, int, int],
    **model_kwargs: Any,
) -> Any:
    config_cls, _ = _lookup(model_name)
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
    config_cls, _ = _lookup(model_name, "checkpoint model")
    return model_name, config_cls(**checkpoint["model_config"])


def serialize_model_config(config: Any) -> dict[str, Any]:
    return asdict(config)
