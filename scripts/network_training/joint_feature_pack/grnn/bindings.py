"""Canonical feature-method names for the independent GRNN workflow."""
from __future__ import annotations

from scripts.network_training.joint_feature_pack.joint.contract import METHODS


def validate_method(method: str) -> str:
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"GRNN method must be one of {', '.join(METHODS)}.")
    return method


def model_name(method: str) -> str:
    return f"{validate_method(method).lower()}_grnn_range"


def artifact_type(method: str) -> str:
    return f"{validate_method(method).lower()}_grnn_reference"
