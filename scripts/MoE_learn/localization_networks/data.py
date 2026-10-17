"""Prepare disjoint learning stages with preprocessing fitted on stage one only."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Subset

from scripts.MoE_learn.data_processing.generation_config import ROOT, normalize_config
from scripts.MoE_learn.data_processing.generation_core import generate_datasets
from scripts.MoE_learn.data_processing.joint_features import build_joint_dataset
from scripts.MoE_learn.localization_networks.common.h5_utils import split_indices, subset_labels
from scripts.MoE_learn.localization_networks.common.input_contract import require_matching_input_signature
from scripts.MoE_learn.localization_networks.rbd_scm.network.data import (
    DatasetBundle, JointRangeH5Dataset, fit_preprocessing,
)


WINDOW_CENTERED_GENERATOR = "moe-learn-python-features-v3-window-centered"


def _generation_caveat(signature: dict) -> str:
    source = signature["source_contract"]
    version = source["generator_version"]
    if version == WINDOW_CENTERED_GENERATOR:
        text = "Feature generator uses per-window per-element mean centering; other windows do not determine that centering."
    else:
        text = f"Generator version {version}; per-window mean centering is not certified by this data contract."
    if source["effective_config"]["dataset"].get("split_strategy") == "range_overlap":
        text += " Range-overlap pool selection uses the source trajectory labels."
    return text


def stage_memberships(dataset, count: int, mode: str, val_fraction: float, seed: int):
    """Partition source rows before per-stage train/validation selection."""
    if type(count) is not int or count < 2 or mode not in {"interleaved", "chronological"}:
        raise ValueError("Stages require a count >= 2 and a supported ordering.")
    order = sorted(range(len(dataset)), key=lambda i: dataset.window_center_s[i])
    if mode == "interleaved":
        groups = [order[i::count] for i in range(count)]
    else:
        groups = [part.tolist() for part in np.array_split(order, count)]
    memberships = []
    for stage, group in enumerate(groups):
        if len(group) < 3:
            raise ValueError("Each stage needs at least three training-source rows before validation splitting.")
        train, val = split_indices(len(group), val_fraction, seed + stage)
        memberships.append(([group[i] for i in train], [group[i] for i in val]))
    flat = [index for train, val in memberships for index in (*train, *val)]
    if len(flat) != len(set(flat)) or set(flat) != set(order):
        raise ValueError("Stage membership must partition each source row exactly once.")
    return memberships


def _arrays(dataset, indices):
    pairs = [dataset[index] for index in indices]
    return (np.stack([x.numpy() for x, _ in pairs]).astype(np.float32),
            np.stack([y.numpy() for _, y in pairs]).astype(np.float32))


def _window_vectors(path: Path) -> dict:
    """Retain raw support intervals, not just center times or source IDs."""
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[f"time/{name}"]).reshape(-1).tolist() for name in
                ("window_start_s", "window_center_s", "window_stop_s", "sample_start_idx", "sample_stop_idx")}


def _require_disjoint_windows(before: dict, after: dict) -> None:
    """Time intervals are half-open; stored sample indices are inclusive."""
    for start, stop, inclusive in (("window_start_s", "window_stop_s", False),
                                    ("sample_start_idx", "sample_stop_idx", True)):
        intervals = list(zip(before[start] + after[start], before[stop] + after[stop]))
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            overlaps = left[1] >= right[0] if inclusive else left[1] > right[0]
            if overlaps:
                raise ValueError(f"before_then_after requires disjoint physical windows: {start}")
        wrong_side = max(before[stop]) >= min(after[start]) if inclusive else max(before[stop]) > min(after[start])
        if wrong_side:
            raise ValueError("before_then_after requires the complete before pool to precede the after pool")


def _ordered(dataset, indices=None):
    selected = range(len(dataset)) if indices is None else indices
    return sorted(selected, key=lambda i: (dataset.window_center_s[i], dataset.source_segment_idx[i]))


def _source_ids(dataset, indices):
    return [dataset.source_segment_idx[index] for index in indices]


def _range_memberships(dataset, train_idx, val_idx, test_idx, edges):
    """Assign fixed roles to predeclared disjoint distance bins, near to far."""
    labels = dataset.labels
    if any(not edges[0] <= label <= edges[-1] for label in labels):
        raise ValueError("after_range_edges_km must cover every after-source label")
    memberships, groups = [], []
    for number, (lower, upper) in enumerate(zip(edges, edges[1:]), 1):
        inclusive = number == len(edges) - 1
        def within(index):
            value = labels[index]
            return lower <= value and (value <= upper if inclusive else value < upper)
        train = [i for i in train_idx if within(i)]
        val = [i for i in val_idx if within(i)]
        positions = [position for position, i in enumerate(test_idx) if within(i)]
        if not train or not val or not positions:
            raise ValueError(f"Range group {number} requires nonempty training, validation and test rows")
        test = [test_idx[position] for position in positions]
        group = {"group": number, "lower_km": lower, "upper_km": upper, "upper_inclusive": inclusive,
                 "train_count": len(train), "val_count": len(val), "test_count": len(test),
                 "train_source_ids": _source_ids(dataset, train), "val_source_ids": _source_ids(dataset, val),
                 "test_source_ids": _source_ids(dataset, test), "test_positions": positions,
                 "test_ranges_km": [labels[i] for i in test]}
        for role, indices in (("train", train), ("val", val), ("test", test)):
            group[f"{role}_range_min_km"] = min(labels[i] for i in indices)
            group[f"{role}_range_max_km"] = max(labels[i] for i in indices)
        memberships.append((train, val))
        groups.append(group)
    for role, original in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        assigned = [source for group in groups for source in group[f"{role}_source_ids"]]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(_source_ids(dataset, original)):
            raise ValueError(f"Distance groups must partition all {role} identities exactly once")
    return memberships, groups


def _stage_from_pool(dataset, path, windows, train_idx, val_idx, number, side, split_mode):
    bundle = DatasetBundle(
        dataset, Subset(dataset, train_idx), Subset(dataset, val_idx), train_idx, val_idx,
        subset_labels(dataset, train_idx), dataset.input_shape, split_mode, [path],
        deepcopy(dataset.input_signature),
    )
    train_x, train_y = _arrays(dataset, bundle.train_idx)
    val_x, val_y = _arrays(dataset, bundle.val_idx)
    stage = {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y}
    identity = {"stage": number, "pool": f"{side}_source", "source_side": side, "source_path": str(path),
                "train_idx": train_idx, "val_idx": val_idx, "train_count": len(train_idx), "val_count": len(val_idx),
                "train_source_ids": _source_ids(dataset, train_idx), "val_source_ids": _source_ids(dataset, val_idx)}
    for role, indices in (("train", train_idx), ("val", val_idx)):
        identity[f"{role}_ranges_km"] = [dataset.labels[index] for index in indices]
        identity.update({f"{role}_{name}": [values[index] for index in indices] for name, values in windows.items()})
    return stage, identity


def _load_before_then_after(config, before, after, before_path, after_path):
    d, training, continual = config["dataset"], config["training"], config["continual"]
    if d["split_strategy"] != "range_overlap" or d.get("train_side") != "before":
        raise ValueError("before_then_after requires range_overlap with train_side=before")
    source_dataset = before.input_signature["source_contract"]["effective_config"]["dataset"]
    if source_dataset.get("split_strategy") != "range_overlap" or source_dataset.get("train_side") != "before":
        raise ValueError("before_then_after requires generated source provenance for range_overlap/train_side=before")
    if before.input_signature["source_contract"]["generator_version"] != WINDOW_CENTERED_GENERATOR:
        raise ValueError("before_then_after requires the per-window-centered generator; regenerate older feature files")
    count = continual["stages"]
    if type(count) is not int or count < 2:
        raise ValueError("before_then_after needs at least two stages")
    if len(before) < 3:
        raise ValueError("Stage one needs at least three before-source rows")
    before_windows, after_windows = _window_vectors(before_path), _window_vectors(after_path)
    _require_disjoint_windows(before_windows, after_windows)
    seed = training["seed"]
    seeds = {"before_validation": seed, "after_test": seed + 1, "after_validation": seed + 2}
    before_order, after_order = _ordered(before), _ordered(after)
    train_positions, val_positions = split_indices(len(before_order), d["val_fraction"], seeds["before_validation"])
    before_train = _ordered(before, [before_order[i] for i in train_positions])
    before_val = _ordered(before, [before_order[i] for i in val_positions])
    remaining_positions, test_positions = split_indices(len(after_order), d["after_test_fraction"], seeds["after_test"])
    after_remaining = _ordered(after, [after_order[i] for i in remaining_positions])
    fixed_test = _ordered(after, [after_order[i] for i in test_positions])
    train_positions, val_positions = split_indices(len(after_remaining), d["val_fraction"], seeds["after_validation"])
    after_train = _ordered(after, [after_remaining[i] for i in train_positions])
    after_val = _ordered(after, [after_remaining[i] for i in val_positions])
    incremental_count = count - 1
    if min(len(after_train), len(after_val)) < incremental_count:
        raise ValueError("Each after increment requires at least one disjoint training and validation row")
    grouping = d["after_grouping"]
    range_groups = []
    if grouping == "range_bins":
        memberships, range_groups = _range_memberships(
            after, after_train, after_val, fixed_test, d["after_range_edges_km"])
        if len(memberships) != incremental_count:
            raise ValueError("Distance groups must match the number of incremental stages")
    elif grouping == "time_interleaved":
        memberships = [(after_train[index::incremental_count], after_val[index::incremental_count])
                       for index in range(incremental_count)]
    else:
        raise ValueError("after_grouping must be time_interleaved or range_bins")
    preprocessing = fit_preprocessing(before, before_train)
    before.set_preprocessing(preprocessing)
    after.set_preprocessing(preprocessing)
    initial, initial_identity = _stage_from_pool(before, before_path, before_windows, before_train, before_val,
                                                  1, "before", d["split_strategy"])
    stages, identities = [initial], [initial_identity]
    for number, (train_idx, val_idx) in enumerate(memberships, 2):
        stage, identity = _stage_from_pool(after, after_path, after_windows, train_idx, val_idx, number,
                                           "after", d["split_strategy"])
        if range_groups:
            identity["range_group"] = deepcopy(range_groups[number - 2])
        stages.append(stage)
        identities.append(identity)
    test_x, test_y = _arrays(after, fixed_test)
    all_ids = [source for stage in identities for role in ("train_source_ids", "val_source_ids") for source in stage[role]]
    test_ids = _source_ids(after, fixed_test)
    expected_ids = set(before.source_segment_idx) | set(after.source_segment_idx)
    if len(set(all_ids + test_ids)) != len(all_ids + test_ids) or set(all_ids + test_ids) != expected_ids:
        raise ValueError("before_then_after must allocate every source to exactly one stage role or fixed test")
    allocation = {
        "policy": ("before-full-after-fixed-test-range-bins-v1" if range_groups
                   else "before-full-after-fixed-test-interleaved-increments-v1"), "seeds": seeds,
        "after_test_fraction": d["after_test_fraction"], "validation_fraction": d["val_fraction"],
        "incremental_groups": incremental_count, "assignment_uses_labels": bool(range_groups),
        "after_grouping": grouping, "range_groups": range_groups,
        "range_group_partition": "PASS" if range_groups else "NOT_APPLICABLE",
        "before": {"pool": "before_source", "source_side": "before", "source_path": str(before_path),
                   "total_count": len(before), "train_count": len(before_train), "val_count": len(before_val),
                   "train_source_ids": _source_ids(before, before_train), "val_source_ids": _source_ids(before, before_val)},
        "after": {"pool": "after_source", "source_side": "after", "source_path": str(after_path),
                  "total_count": len(after), "train_count": len(after_train), "val_count": len(after_val), "test_count": len(fixed_test),
                  "train_source_ids": _source_ids(after, after_train), "val_source_ids": _source_ids(after, after_val), "test_source_ids": test_ids},
        "source_identity_partition": "PASS", "physical_window_partition": "PASS",
        "before_after_temporal_separation": "PASS", "test_is_fixed_across_stages": True,
    }
    metadata = {
        "pack": "MoE_learn", "method": "RBD_SCM", "fusion": "block_rms",
        "input_signature": deepcopy(before.input_signature), "input_shape": list(before.input_shape),
        "source_paths": [str(before_path), str(after_path)], "preprocessing": preprocessing,
        "preprocessing_fit_stage": 1, "input_normalized_per_sample": True, "stage_mode": "before_then_after",
        "split_mode": d["split_strategy"], "stages": identities, "allocation": allocation,
        "test_pool": "after_source", "test_source_side": "after", "test_source_path": str(after_path),
        "test_idx": fixed_test, "test_count": len(fixed_test), "test_source_ids": test_ids,
        **{f"test_{name}": [values[index] for index in fixed_test] for name, values in after_windows.items()},
        "evidence_scope": "single-record before-to-after staged adaptation with a fixed disjoint after-side test set; not unseen-environment evidence",
        "generation_caveat": _generation_caveat(before.input_signature),
    }
    return {"stages": stages, "test_x": test_x, "test_y": test_y, "metadata": metadata}


def _distance_partitions(dataset, edges):
    if (len(edges) < 3 or not all(np.isfinite(value) and value > 0 for value in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("Distance holdout requires finite positive increasing edges")
    if any(not edges[0] <= value <= edges[-1] for value in dataset.labels):
        raise ValueError("distance_edges_km must cover every before and after label")
    groups = []
    for number, (lower, upper) in enumerate(zip(edges, edges[1:]), 1):
        groups.append(_ordered(dataset, [i for i, value in enumerate(dataset.labels)
            if lower <= value and (value <= upper if number == len(edges) - 1 else value < upper)]))
    return groups


def _load_distance_holdout(config, before, after, before_path, after_path):
    """Exclude one complete distance interval from learning on both sides."""
    d, t, continual = config["dataset"], config["training"], config["continual"]
    source = before.input_signature["source_contract"]
    if (d["split_strategy"] != "range_overlap" or d["train_side"] != "before"
            or source["effective_config"]["dataset"].get("split_strategy") != "range_overlap"
            or source["effective_config"]["dataset"].get("train_side") != "before"
            or source["generator_version"] != WINDOW_CENTERED_GENERATOR):
        raise ValueError("distance_holdout requires window-centered range_overlap/train_side=before sources")
    edges, held = d["distance_edges_km"], d["heldout_group"]
    if type(held) is not int or not 1 <= held < len(edges) or continual["stages"] != len(edges) - 1:
        raise ValueError("Distance holdout group and stage count do not match the edges")
    before_windows, after_windows = _window_vectors(before_path), _window_vectors(after_path)
    _require_disjoint_windows(before_windows, after_windows)
    before_groups, after_groups = _distance_partitions(before, edges), _distance_partitions(after, edges)
    excluded_before, fixed_test = before_groups[held - 1], after_groups[held - 1]
    if not fixed_test:
        raise ValueError("Held-out after distance group must contain test rows")
    allowed = [number for number in range(1, len(edges)) if number != held]
    before_pool = _ordered(before, [i for number in allowed for i in before_groups[number - 1]])
    after_pool = _ordered(after, [i for number in allowed for i in after_groups[number - 1]])
    if len(before_pool) < 3 or len(after_pool) < 3:
        raise ValueError("Distance holdout leaves insufficient before/after learning rows")
    seeds = {"before_validation": t["seed"], "after_validation": t["seed"] + 2}
    def split_pool(dataset, pool, seed):
        train, val = split_indices(len(pool), d["val_fraction"], seed)
        return (_ordered(dataset, [pool[i] for i in train]), _ordered(dataset, [pool[i] for i in val]))
    before_train, before_val = split_pool(before, before_pool, seeds["before_validation"])
    after_train, after_val = split_pool(after, after_pool, seeds["after_validation"])
    preprocessing = fit_preprocessing(before, before_train)
    before.set_preprocessing(preprocessing)
    after.set_preprocessing(preprocessing)
    initial, initial_identity = _stage_from_pool(before, before_path, before_windows, before_train, before_val,
                                                  1, "before", d["split_strategy"])
    stages, identities, groups = [initial], [initial_identity], []
    for number, (lower, upper) in enumerate(zip(edges, edges[1:]), 1):
        group_ids = set(after_groups[number - 1])
        train = [i for i in after_train if i in group_ids]
        val = [i for i in after_val if i in group_ids]
        group = {"group": number, "lower_km": lower, "upper_km": upper, "upper_inclusive": number == len(edges) - 1,
                 "heldout": number == held, "before_count": len(before_groups[number - 1]),
                 "after_count": len(group_ids), "train_count": len(train), "val_count": len(val),
                 "test_count": len(fixed_test) if number == held else 0,
                 "before_source_ids": _source_ids(before, before_groups[number - 1]),
                 "after_source_ids": _source_ids(after, after_groups[number - 1]),
                 "before_ranges_km": [before.labels[i] for i in before_groups[number - 1]],
                 "after_ranges_km": [after.labels[i] for i in after_groups[number - 1]]}
        groups.append(group)
        if number == held:
            continue
        if not train or not val:
            raise ValueError(f"Learning distance group {number} requires nonempty training and validation rows")
        stage, identity = _stage_from_pool(after, after_path, after_windows, train, val,
                                           len(stages) + 1, "after", d["split_strategy"])
        identity["range_group"] = deepcopy(group)
        stages.append(stage)
        identities.append(identity)
    test_x, test_y = _arrays(after, fixed_test)
    test_ids, excluded_ids = _source_ids(after, fixed_test), _source_ids(before, excluded_before)
    learning_ids = [source for stage in identities for role in ("train_source_ids", "val_source_ids") for source in stage[role]]
    allocated = learning_ids + test_ids + excluded_ids
    if len(set(allocated)) != len(allocated) or set(allocated) != set(before.source_segment_idx + after.source_segment_idx):
        raise ValueError("Holdout learning, test and excluded-before identities must partition every source")
    lower, upper = edges[held - 1:held + 1]
    for identity in identities:
        for role in ("train", "val"):
            if any(lower <= value and (value <= upper if held == len(edges) - 1 else value < upper)
                   for value in identity[f"{role}_ranges_km"]):
                raise ValueError("Held-out test distance leaked into learning")
    allocation = {"policy": "before-after-distance-holdout-v1", "heldout_group": held,
        "distance_edges_km": list(edges), "distance_groups": groups, "seeds": seeds,
        "validation_fraction": d["val_fraction"], "assignment_uses_labels": True,
        "test_range_excluded_from_learning": "PASS", "source_identity_partition": "PASS",
        "physical_window_partition": "PASS", "before_after_temporal_separation": "PASS",
        "test_is_fixed_across_stages": True,
        "before": {"total_count": len(before), "train_count": len(before_train), "val_count": len(before_val),
                   "excluded_count": len(excluded_ids), "excluded_source_ids": excluded_ids,
                   "excluded_ranges_km": [before.labels[i] for i in excluded_before],
                   "train_source_ids": _source_ids(before, before_train), "val_source_ids": _source_ids(before, before_val)},
        "after": {"total_count": len(after), "train_count": len(after_train), "val_count": len(after_val),
                  "test_count": len(fixed_test), "train_source_ids": _source_ids(after, after_train),
                  "val_source_ids": _source_ids(after, after_val), "test_source_ids": test_ids}}
    metadata = {"pack": "MoE_learn", "method": "RBD_SCM", "fusion": "block_rms",
        "input_signature": deepcopy(before.input_signature), "input_shape": list(before.input_shape),
        "source_paths": [str(before_path), str(after_path)], "preprocessing": preprocessing,
        "preprocessing_fit_stage": 1, "input_normalized_per_sample": True, "stage_mode": "distance_holdout",
        "split_mode": d["split_strategy"], "stages": identities, "allocation": allocation,
        "test_pool": "after_heldout_distance", "test_source_side": "after", "test_source_path": str(after_path),
        "test_idx": fixed_test, "test_count": len(fixed_test), "test_source_ids": test_ids,
        "test_ranges_km": [after.labels[i] for i in fixed_test],
        **{f"test_{name}": [values[i] for i in fixed_test] for name, values in after_windows.items()},
        "evidence_scope": "same-record held-out distance generalization; test interval excluded from before/after training, validation and replay; not an independent environment",
        "generation_caveat": _generation_caveat(before.input_signature)}
    return {"stages": stages, "test_x": test_x, "test_y": test_y, "metadata": metadata}


def load_prepared_data(config: dict, train_path: Path, test_path: Path) -> dict:
    """Load owned HDF5 files, checking axes, split identity and frozen scales."""
    train = JointRangeH5Dataset([train_path], normalize_input=True)
    test = JointRangeH5Dataset([test_path], normalize_input=True)
    try:
        require_matching_input_signature(train.input_signature, test.input_signature, "Demo train/test")
        if set(train.source_segment_idx) & set(test.source_segment_idx):
            raise ValueError("Demo train/test source identities overlap.")
        d, t, c = config["dataset"], config["training"], config["continual"]
        if d["stage_mode"] == "distance_holdout":
            return _load_distance_holdout(config, train, test, train_path, test_path)
        if d["stage_mode"] == "before_then_after":
            return _load_before_then_after(config, train, test, train_path, test_path)
        memberships = stage_memberships(train, c["stages"], d["stage_mode"], d["val_fraction"], t["seed"])
        preprocessing = fit_preprocessing(train, memberships[0][0])
        train.set_preprocessing(preprocessing)
        test.set_preprocessing(preprocessing)
        stages, identities = [], []
        for number, (train_idx, val_idx) in enumerate(memberships, 1):
            bundle = DatasetBundle(
                train, Subset(train, train_idx), Subset(train, val_idx), train_idx, val_idx,
                subset_labels(train, train_idx), train.input_shape, d["split_strategy"],
                [train_path], deepcopy(train.input_signature),
            )
            train_x, train_y = _arrays(train, bundle.train_idx)
            val_x, val_y = _arrays(train, bundle.val_idx)
            stages.append({"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y})
            identities.append({
                "stage": number, "train_idx": bundle.train_idx, "val_idx": bundle.val_idx,
                "train_source_ids": [train.source_segment_idx[i] for i in train_idx],
                "val_source_ids": [train.source_segment_idx[i] for i in val_idx],
                "train_window_center_s": [train.window_center_s[i] for i in train_idx],
                "val_window_center_s": [train.window_center_s[i] for i in val_idx],
            })
        test_x, test_y = _arrays(test, list(range(len(test))))
        metadata = {
            "pack": "MoE_learn", "method": "RBD_SCM", "fusion": "block_rms",
            "input_signature": deepcopy(train.input_signature), "input_shape": list(train.input_shape),
            "source_paths": [str(train_path), str(test_path)],
            "preprocessing": preprocessing, "preprocessing_fit_stage": 1,
            "input_normalized_per_sample": True, "stage_mode": d["stage_mode"],
            "split_mode": d["split_strategy"], "stages": identities,
            "test_source_ids": list(test.source_segment_idx),
            "test_window_center_s": list(test.window_center_s),
            "evidence_scope": "single-record small demo; not an unseen-environment experiment",
            "generation_caveat": _generation_caveat(train.input_signature),
        }
        return {"stages": stages, "test_x": test_x, "test_y": test_y, "metadata": metadata}
    finally:
        train.close()
        test.close()


def load_demo_data(config: dict, output_dir: Path) -> dict:
    """Generate a private real-data demo inside the current frozen snapshot."""
    dataset = {key: deepcopy(value) for key, value in config["dataset"].items()
               if key not in {"stage_mode", "val_fraction", "after_test_fraction", "after_grouping", "after_range_edges_km", "distance_edges_km", "heldout_group"}}
    generation = normalize_config({"schema_version": 1, "dataset": dataset,
                                   "rbd": deepcopy(config["rbd"]),
                                   "result_path": "outputs/MoE_learn/generation-result.json"})
    print("Generating aligned RBD and SCM features from the configured S5 interval.", flush=True)
    result = generate_datasets(generation)
    joint = build_joint_dataset(result["datasets"], ROOT / "outputs" / "MoE_learn" / "joint")
    directory = Path(joint["RBD_SCM"])
    prepared = load_prepared_data(config, directory / "RBD_SCM_train.h5", directory / "RBD_SCM_test.h5")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data-metadata.json").write_text(
        json.dumps(prepared["metadata"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(prepared['stages'])} disjoint stages; test rows: {len(prepared['test_y'])}.", flush=True)
    return prepared
