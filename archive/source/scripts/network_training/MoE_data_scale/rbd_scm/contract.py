"""Training-fitted preprocessing contract for MoE_data_scale RBD_SCM."""
from __future__ import annotations

import math

from scripts.data_processing.MoE_data_scale.contract import (
    FORMAT_VERSION,
    FUSION_STRATEGIES,
    METHODS,
    components,
    validate_signature,
)


def validate_preprocessing(state: dict, signature: dict) -> None:
    if not isinstance(state, dict) or set(state) != {"strategy", "scales", "fit_sample_count", "fit_source_identity_sha256"}:
        raise ValueError("Joint preprocessing must contain strategy, scales, fit_sample_count and fit_source_identity_sha256.")
    if state["strategy"] != signature["fusion_strategy"]:
        raise ValueError("Joint preprocessing strategy differs from dataset signature.")
    scales = state["scales"]
    if not isinstance(scales, list) or len(scales) != len(signature["blocks"]):
        raise ValueError("Joint preprocessing scale count differs from feature blocks.")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Joint preprocessing scales must be finite and positive.")
    if type(state["fit_sample_count"]) is not int or state["fit_sample_count"] <= 0:
        raise ValueError("Joint preprocessing requires a positive fit sample count.")
    digest = state["fit_source_identity_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Joint preprocessing requires a retained-training source identity SHA256.")


__all__ = [
    "FORMAT_VERSION",
    "FUSION_STRATEGIES",
    "METHODS",
    "components",
    "validate_preprocessing",
    "validate_signature",
]
