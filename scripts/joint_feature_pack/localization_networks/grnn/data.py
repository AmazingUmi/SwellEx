"""Raw CHW blocks for joint GRNN; each fold fits its own scales later."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from scripts.joint_feature_pack.localization_networks.grnn.bindings import validate_method
from scripts.joint_feature_pack.localization_networks.joint.data import JointRangeH5Dataset


@dataclass
class GrnnFeatureBundle:
    x: torch.Tensor
    y_range_km: torch.Tensor
    source_segment_idx: list[int]
    window_center_s: list[float]
    input_shape: tuple[int, int, int]
    source_paths: list[Path]
    input_signature: dict[str, object]


def load_grnn_features(h5_paths: Iterable[Path], *, method: str,
                       per_sample_channel_norm: bool = False,
                       max_samples: int | None = None) -> GrnnFeatureBundle:
    method = validate_method(method)
    if per_sample_channel_norm is not False:
        raise ValueError("Joint GRNN uses fusion preprocessing only; per-sample channel normalization is unsupported.")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be a positive integer.")
    paths = [Path(path) for path in h5_paths]
    dataset = JointRangeH5Dataset(paths, normalize_input=False, method=method)
    try:
        count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        return GrnnFeatureBundle(
            x=torch.stack([dataset[index][0].reshape(-1) for index in range(count)]),
            y_range_km=torch.tensor(dataset.labels[:count], dtype=torch.float32).view(-1, 1),
            source_segment_idx=dataset.source_segment_idx[:count], window_center_s=dataset.window_center_s[:count],
            input_shape=dataset.input_shape, source_paths=paths, input_signature=dataset.input_signature,
        )
    finally:
        dataset.close()
