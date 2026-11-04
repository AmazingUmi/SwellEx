"""Materialize aligned RBD_SCM feature combinations without fitting preprocessing scales."""
from __future__ import annotations

from pathlib import Path

from scripts.data_processing.common.joint_builder import ALIGNMENT_FIELDS, build_aligned_joint_datasets
from scripts.data_processing.MoE_data_scale.contract import (
    FORMAT_VERSION,
    FUSION_STRATEGIES,
    METHODS,
    components,
    validate_signature,
)

ROUTES = ("RBD", "SCM")


def build_joint_dataset(dataset_dirs: dict[str, str | Path], output_dir: Path) -> dict[str, str]:
    return build_aligned_joint_datasets(
        dataset_dirs,
        output_dir,
        pack="MoE_data_scale",
        routes=ROUTES,
        methods=METHODS,
        fusion_strategy="block_rms",
        allowed_fusions=FUSION_STRATEGIES,
        validate_signature_fn=validate_signature,
        require_exact_routes=False,
    )


__all__ = ["ALIGNMENT_FIELDS", "FORMAT_VERSION", "FUSION_STRATEGIES", "METHODS", "ROUTES", "build_joint_dataset", "components"]
