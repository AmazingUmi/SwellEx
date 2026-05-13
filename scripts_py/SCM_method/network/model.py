"""SCM model registry and construction helpers."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from torch import nn

from .models.model_scm_complex_cnn_range import (
    ModelConfig as ScmComplexRangeCNNConfig,
)
from .models.model_scm_complex_cnn_range import ScmComplexRangeCNN
from .models.model_scm_real_cnn_range import ModelConfig as ScmRealRangeCNNConfig
from .models.model_scm_real_cnn_range import ScmRealRangeCNN
from .models.model_scm_resnet_range import ModelConfig as ScmResNetRangeConfig
from .models.model_scm_resnet_range import ScmResNet18Range, ScmResNet50Range


MODEL_REGISTRY = {
    "scm_complex_cnn_range": (ScmComplexRangeCNNConfig, ScmComplexRangeCNN),
    "scm_real_cnn_range": (ScmRealRangeCNNConfig, ScmRealRangeCNN),
    "scm_resnet18_range": (ScmResNetRangeConfig, ScmResNet18Range),
    "scm_resnet50_range": (ScmResNetRangeConfig, ScmResNet50Range),
}

MODEL_DATASET_INPUT_NORM = {
    "scm_complex_cnn_range": True,
    "scm_real_cnn_range": False,
    "scm_resnet18_range": True,
    "scm_resnet50_range": True,
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

    _, pairs, freq_bins = input_shape
    config_values = {
        "input_pairs": pairs,
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
    model_name = str(checkpoint.get("model_name", "scm_complex_cnn_range"))
    try:
        config_cls, _ = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint model: {model_name}") from exc
    return model_name, config_cls(**checkpoint["model_config"])


def serialize_model_config(config: Any) -> dict[str, Any]:
    return asdict(config)
