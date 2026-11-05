"""Canonical joint feature schema for joint_feature_pack."""
from __future__ import annotations

from scripts.data_processing.common.joint_contract import (
    FORMAT_VERSION,
    split_components,
    validate_joint_signature,
)

METHODS = ("RBD", "ELM", "SCM", "RBD_ELM", "RBD_SCM", "ELM_SCM", "RBD_ELM_SCM")
FUSION_STRATEGIES = ("raw_concat", "block_rms")


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
