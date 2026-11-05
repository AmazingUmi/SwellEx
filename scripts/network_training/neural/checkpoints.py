"""Strict contract for neural artifacts written by the current runtime."""
from __future__ import annotations

import math
from typing import Any


CHECKPOINT_VERSION = 1


def validate_checkpoint(checkpoint: dict[str, Any], *, method: str, model_name: str, dataset_name: str) -> None:
    required = {"checkpoint_version", "method", "model_name", "dataset_name", "model_config",
                "model_state", "input_signature", "input_normalized_per_sample",
                "target_mean_km", "target_std_km", "loss_space", "huber_beta"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        missing = sorted(required - checkpoint.keys()) if isinstance(checkpoint, dict) else sorted(required)
        raise ValueError(f"Unsupported checkpoint: missing contract fields {missing}; regenerate with current scripts.")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version; regenerate with current scripts.")
    for key, expected in (("method", method), ("model_name", model_name), ("dataset_name", dataset_name)):
        if checkpoint[key] != expected:
            raise ValueError(f"Checkpoint {key} '{checkpoint[key]}' does not match requested '{expected}'.")
    if not isinstance(checkpoint["input_signature"], dict) or checkpoint["input_signature"].get("method") != method:
        raise ValueError("Checkpoint input_signature is missing or inconsistent with its method.")
    if not isinstance(checkpoint["input_normalized_per_sample"], bool):
        raise ValueError("Checkpoint input normalization policy must be boolean.")
    for key in ("target_mean_km", "target_std_km", "huber_beta"):
        if not math.isfinite(float(checkpoint[key])):
            raise ValueError(f"Checkpoint {key} must be finite.")
    if float(checkpoint["target_std_km"]) <= 0 or float(checkpoint["huber_beta"]) < 0:
        raise ValueError("Checkpoint target std must be positive and Huber beta nonnegative.")
    if checkpoint["loss_space"] not in {"km", "normalized"}:
        raise ValueError("Checkpoint loss_space is invalid.")
