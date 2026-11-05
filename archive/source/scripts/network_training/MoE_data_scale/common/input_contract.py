"""Semantic identities of feature axes, shared with data_processing."""
from __future__ import annotations

from typing import Any

from scripts.data_processing.common.input_contract import (
    read_input_signature,
    require_matching_input_signature,
    validate_saved_input_signature as _validate_single_route_signature,
)
from scripts.data_processing.MoE_data_scale.contract import validate_signature as _validate_joint_signature


def validate_saved_input_signature(signature: Any, input_shape: tuple[int, int, int], method: str) -> None:
    """Reject incomplete physical axes before a saved model can be restored."""
    if method == "RBD_SCM":
        _validate_joint_signature(signature, input_shape, method)
        return
    _validate_single_route_signature(signature, input_shape, method)


__all__ = [
    "read_input_signature",
    "require_matching_input_signature",
    "validate_saved_input_signature",
]
