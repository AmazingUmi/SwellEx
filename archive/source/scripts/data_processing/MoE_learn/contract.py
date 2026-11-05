"""Canonical RBD_SCM joint feature schema for MoE_learn."""
from __future__ import annotations

from scripts.data_processing.common.joint_contract import (
    FORMAT_VERSION,
    split_components,
    validate_joint_signature,
)

METHODS = ("RBD_SCM",)
FUSION_STRATEGIES = ("block_rms",)


def components(method: str) -> tuple[str, ...]:
    return split_components(method, METHODS)


def validate_signature(signature: dict, shape: tuple[int, int, int], method: str | None = None) -> None:
    validate_joint_signature(
        signature,
        shape,
        method,
        allowed_methods=METHODS,
        allowed_fusions=FUSION_STRATEGIES,
    )
