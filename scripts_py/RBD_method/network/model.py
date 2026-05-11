"""Model registry and construction helpers."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from torch import nn

from .models.model_complex_cnn_range import (
    ComplexRangeCNN,
    ModelConfig as ComplexRangeCNNConfig,
)
from .models.model_real_cnn_range import (
    ModelConfig as RealRangeCNNConfig,
    RealRangeCNN,
)
from .models.model_resnet_range import (
    ModelConfig as ResNetRangeConfig,
    ResNet18Range,
    ResNet50Range,
)


MODEL_REGISTRY = {
    "complex_cnn_range": (ComplexRangeCNNConfig, ComplexRangeCNN),
    "real_cnn_range": (RealRangeCNNConfig, RealRangeCNN),
    "resnet18_range": (ResNetRangeConfig, ResNet18Range),
    "resnet50_range": (ResNetRangeConfig, ResNet50Range),
}

MODEL_DATASET_INPUT_NORM = {
    "complex_cnn_range": True,
    "real_cnn_range": False,
    "resnet18_range": True,
    "resnet50_range": True,
}


def build_model_config(
    model_name: str,
    input_shape: tuple[int, int, int],
    **model_kwargs: Any,
) -> Any:
    try:
        config_cls, _ = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

    _, elements, freq_bins = input_shape
    config_values = {
        "input_elements": elements,
        "input_freq_bins": freq_bins,
    }
    config_fields = {field.name for field in fields(config_cls)}
    config_values.update(
        {
            key: value
            for key, value in model_kwargs.items()
            if key in config_fields and value is not None
        }
    )
    return config_cls(**config_values)


def build_model(model_name: str, config: Any) -> nn.Module:
    try:
        _, model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    return model_cls(config)


def model_uses_dataset_input_norm(model_name: str) -> bool:
    try:
        return MODEL_DATASET_INPUT_NORM[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[str, Any]:
    model_name = str(checkpoint.get("model_name", "complex_cnn_range"))
    try:
        config_cls, _ = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint model: {model_name}") from exc
    return model_name, config_cls(**checkpoint["model_config"])


def serialize_model_config(config: Any) -> dict[str, Any]:
    return asdict(config)
