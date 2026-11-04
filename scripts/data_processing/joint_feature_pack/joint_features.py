"""Materialize aligned 7-way feature combinations without fitting preprocessing scales."""
from __future__ import annotations

from pathlib import Path

from scripts.data_processing.common.joint_builder import ALIGNMENT_FIELDS, build_aligned_joint_datasets
from scripts.data_processing.joint_feature_pack.contract import (
    FORMAT_VERSION,
    FUSION_STRATEGIES,
    METHODS,
    components,
    validate_signature,
)

ROUTES = ("RBD", "ELM", "SCM")


def build_joint_datasets(dataset_dirs: dict[str, str | Path], output_root: Path, *,
                         fusion_strategy: str = "block_rms") -> dict[str, str]:
    return build_aligned_joint_datasets(
        dataset_dirs,
        output_root,
        pack="joint_feature_pack",
        routes=ROUTES,
        methods=METHODS,
        fusion_strategy=fusion_strategy,
        allowed_fusions=FUSION_STRATEGIES,
        validate_signature_fn=validate_signature,
        require_exact_routes=True,
    )


__all__ = ["ALIGNMENT_FIELDS", "FORMAT_VERSION", "FUSION_STRATEGIES", "METHODS", "ROUTES", "build_joint_datasets", "components"]
