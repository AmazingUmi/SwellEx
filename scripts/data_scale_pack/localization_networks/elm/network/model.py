"""Strict ELM registry for pre-training complex CNN capacity selection."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from torch import nn

from scripts.data_scale_pack.localization_networks.common.scaling import (
    positive_int,
    resolution_metadata,
    resolve_capacity,
)
from .models.complex_cnn import ModelConfig, ElmComplexRangeCNN


_MODEL_PROFILES = {
    "elm_complex_cnn_range": "auto",
    "elm_complex_cnn_small_range": "small",
    "elm_complex_cnn_medium_range": "medium",
    "elm_complex_cnn_large_range": "large",
    "elm_complex_cnn_xlarge_range": "xlarge",
    "elm_complex_cnn_baseline_range": "baseline",
}
MODEL_REGISTRY = {name: (ModelConfig, ElmComplexRangeCNN) for name in _MODEL_PROFILES}
MODEL_DATASET_INPUT_NORM = {name: True for name in MODEL_REGISTRY}


def build_model_config(
    model_name: str,
    input_shape: tuple[int, int, int],
    **model_kwargs: Any,
) -> ModelConfig:
    try:
        profile = _MODEL_PROFILES[model_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError("input_shape must contain real/imag channels, feature rows, frequency bins")
    if type(input_shape[0]) is not int or input_shape[0] != 2:
        raise ValueError("input_shape must use two real/imaginary channels")
    positive_int(input_shape[1], "input_pairs")
    positive_int(input_shape[2], "input_freq_bins")
    allowed = {"base_channels", "dropout", "train_count", "scaling_policy"}
    unknown = set(model_kwargs) - allowed
    if unknown:
        raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
    missing = {"train_count", "scaling_policy"} - set(model_kwargs)
    if missing:
        raise ValueError(f"Missing model configuration fields: {sorted(missing)}")
    base_channels = model_kwargs.get("base_channels", 16)
    resolved = resolve_capacity(
        profile,
        train_count=model_kwargs["train_count"],
        base_channels=base_channels,
        scaling_policy=model_kwargs["scaling_policy"],
    )
    return ModelConfig(
        input_pairs=input_shape[1],
        input_freq_bins=input_shape[2],
        base_channels=base_channels,
        dropout=model_kwargs.get("dropout", 0.15),
        **resolved,
    )


def _validate_identity(model_name: str, config: Any) -> None:
    try:
        profile = _MODEL_PROFILES[model_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc
    if not isinstance(config, ModelConfig):
        raise ValueError("Model configuration belongs to a different method or schema")
    config.__post_init__()
    if profile == "auto":
        if config.selection_mode != "auto":
            raise ValueError("Automatic model requires saved automatic selection")
    elif config.selection_mode != "fixed" or config.selected_profile != profile:
        raise ValueError("Fixed model identifier disagrees with saved capacity profile")


def build_model(model_name: str, config: Any) -> nn.Module:
    _validate_identity(model_name, config)
    return MODEL_REGISTRY[model_name][1](config)


def model_uses_dataset_input_norm(model_name: str) -> bool:
    try:
        return MODEL_DATASET_INPUT_NORM[model_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[str, ModelConfig]:
    try:
        model_name = checkpoint["model_name"]
        config_cls, _ = MODEL_REGISTRY[model_name]
        config = config_cls(**checkpoint["model_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint requires a known model and complete canonical model_config") from exc
    _validate_identity(model_name, config)
    return model_name, config


def serialize_model_config(config: ModelConfig) -> dict[str, Any]:
    if not isinstance(config, ModelConfig):
        raise ValueError("Model configuration belongs to a different method or schema")
    config.__post_init__()
    return json.loads(json.dumps(asdict(config), allow_nan=False))
