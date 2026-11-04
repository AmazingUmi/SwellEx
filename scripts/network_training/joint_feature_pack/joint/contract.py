"""Training-fitted preprocessing and continuation contracts for joint_feature_pack."""
from __future__ import annotations

import math
from pathlib import Path

from scripts.data_processing.joint_feature_pack.contract import (
    FORMAT_VERSION,
    FUSION_STRATEGIES,
    METHODS,
    components,
    validate_signature,
)


def validate_preprocessing(state: dict, signature: dict) -> None:
    if not isinstance(state, dict) or set(state) != {"strategy", "scales", "fit_sample_count"}:
        raise ValueError("Joint preprocessing must contain strategy, scales and fit_sample_count.")
    if state["strategy"] != signature["fusion_strategy"]:
        raise ValueError("Joint preprocessing strategy differs from dataset signature.")
    scales = state["scales"]
    if not isinstance(scales, list) or len(scales) != len(signature["blocks"]):
        raise ValueError("Joint preprocessing scale count differs from feature blocks.")
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Joint preprocessing scales must be finite and positive.")
    if type(state["fit_sample_count"]) is not int or state["fit_sample_count"] <= 0:
        raise ValueError("Joint preprocessing requires a positive fit sample count.")
    if state["strategy"] == "raw_concat" and any(s != 1.0 for s in scales):
        raise ValueError("raw_concat requires unit scales.")


def validate_resume_data_identity(identity: dict) -> None:
    """Validate saved training-source identities without reading training files.

    Prediction may run after the training sources have moved or been removed;
    only continuation compares these identities with current train/val files.
    """
    if (not isinstance(identity, dict)
            or set(identity) != {"format_version", "train_sources", "val_sources"}
            or type(identity["format_version"]) is not int or identity["format_version"] != 1):
        raise ValueError("Joint checkpoint requires a complete resume_data_identity contract.")
    for role in ("train_sources", "val_sources"):
        sources = identity[role]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"resume_data_identity {role} must be a nonempty ordered list.")
        paths = []
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"path", "size_bytes", "sha256"}:
                raise ValueError("Resume source identity must contain path, size_bytes and sha256.")
            if (not isinstance(source["path"], str) or not Path(source["path"]).is_absolute()
                    or ".." in Path(source["path"]).parts
                    or type(source["size_bytes"]) is not int or source["size_bytes"] <= 0
                    or not isinstance(source["sha256"], str) or len(source["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in source["sha256"])):
                raise ValueError("Invalid resume_data_identity source path, size or SHA256.")
            paths.append(source["path"])
        if len(set(paths)) != len(paths):
            raise ValueError(f"resume_data_identity {role} contains duplicate source paths.")


__all__ = [
    "FORMAT_VERSION",
    "FUSION_STRATEGIES",
    "METHODS",
    "components",
    "validate_preprocessing",
    "validate_resume_data_identity",
    "validate_signature",
]
