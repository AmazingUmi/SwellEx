"""Training-only sample budgets and immutable source identity for scaling runs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import random

import torch
from torch.utils.data import Subset

SAMPLING_VERSION = "temporal-block-prefix-v1"
BLOCK_ROWS = 32


def sample_indices(indices, source_ids, times, fraction: float, seed: int):
    """Nested training subsets; validation membership never enters selection.

    Order training candidates by physical time, group 32 consecutive candidates,
    shuffle whole blocks once, and take a prefix. Only the final block can be
    partial. These are sampling blocks, not independent observations or a new
    train/validation separation guarantee.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_fraction must be finite and in (0, 1]")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Training indices must be nonempty and unique")
    if len(source_ids) != len(times) or len(set(source_ids)) != len(source_ids):
        raise ValueError("S5 source-window identities must be unique")
    if any(not math.isfinite(float(t)) for t in times):
        raise ValueError("Sample times must be finite")
    count = math.floor(len(indices) * fraction)
    if count < 2:
        raise ValueError("train_fraction must retain at least two training samples")
    if fraction == 1:
        return list(indices)
    ordered = sorted(indices, key=lambda i: (times[i], source_ids[i]))
    blocks = [ordered[i:i + BLOCK_ROWS] for i in range(0, len(ordered), BLOCK_ROWS)]
    random.Random(seed).shuffle(blocks)
    return sorted([i for block in blocks for i in block][:count])


def select_training_subset(bundle, fraction: float, seed: int):
    dataset = bundle.dataset
    # The current generation contract is S5 only, so a source ID identifies a
    # physical window across files. Reject duplicates instead of inflating N.
    source_ids, times = dataset.source_segment_idx, dataset.window_center_s
    selected = sample_indices(bundle.train_idx, source_ids, times, fraction, seed)
    if bundle.val_dataset is not dataset and not isinstance(bundle.val_dataset, Subset):
        validation_ids = bundle.val_dataset.source_segment_idx
        if len(set(validation_ids)) != len(validation_ids) or set(source_ids) & set(validation_ids):
            raise ValueError("Explicit training/validation S5 source identities overlap or repeat")
    metadata = {
        "version": SAMPLING_VERSION, "train_fraction": float(fraction),
        "candidate_train_count": len(bundle.train_idx), "train_count": len(selected),
        "validation_count": len(bundle.val_idx), "block_rows": BLOCK_ROWS, "seed": seed,
        "source_identity_sha256": hashlib.sha256(
            ",".join(str(source_ids[i]) for i in selected).encode()).hexdigest(),
    }
    labels = torch.tensor([dataset.labels[i] for i in selected], dtype=torch.float32).view(-1, 1)
    return replace(bundle, train_dataset=Subset(dataset, selected), train_idx=selected,
                   train_labels=labels), metadata


def source_identity(train_paths, val_paths):
    """Hash complete HDF5 bytes so resume cannot reuse a modified input file."""
    cache = {}

    def inventory(paths):
        result = []
        for value in paths:
            path = Path(value).resolve(strict=True)
            if path not in cache:
                before = path.stat()
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError("Training source changed while hashing")
                cache[path] = {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}
            result.append(dict(cache[path]))
        return result

    return {"version": 1, "train_sources": inventory(train_paths),
            "validation_sources": inventory(val_paths or train_paths)}
