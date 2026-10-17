"""Strict configuration shared by local dispatch and remote execution."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re
import tomllib

FIELDS = {
    "remote": {"host", "repository", "workspace", "base_python", "uv"},
    "validation": {"device", "seed", "threads", "quick_timeout_s", "generation_timeout_s", "training_timeout_s"},
    "dataset": {"event_name", "fs", "segment_start_s", "segment_end_s", "segment_duration_s", "segment_step_s", "num_snapshots_per_segment", "snapshot_overlap_count", "frequency_selection_modes", "split_strategy", "train_test_ratio", "train_side", "half_duration_s", "gap_s", "test_step_s", "manual_dataset_variant_tag", "stage_mode", "val_fraction"},
    "rbd": {"beam_selection", "frequency_estimation", "use_plane_wave", "normalize_spectrum"},
    "training": {"initial_epochs", "adapt_epochs", "warmup_epochs", "batch_size", "base_channels", "learning_rate", "old_expert_learning_rate", "new_expert_learning_rate", "router_learning_rate", "seed", "weight_decay"},
    "continual": {"stages", "initial_experts", "max_experts", "replay_mode", "replay_size", "history_validation"},
    "controller": {"min_relative_gain", "max_old_relative_regression", "novelty_threshold", "balance_loss_weight"},
}

def positive(value: object, name: str, *, integer: bool = False, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def validate_config(config: dict) -> dict:
    required_tables = {"schema_version", *FIELDS}
    if not isinstance(config, dict) or set(config) != required_tables:
        raise ValueError("Configuration must contain schema_version and exactly the documented tables")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for table, fields in FIELDS.items():
        if table == "dataset" and isinstance(config[table], dict) and config[table].get("stage_mode") == "before_then_after":
            fields = fields | {"after_test_fraction", "after_grouping"}
            if config[table].get("after_grouping") == "range_bins":
                fields = fields | {"after_range_edges_km"}
        if table == "dataset" and isinstance(config[table], dict) and config[table].get("stage_mode") == "distance_holdout":
            fields = fields | {"distance_edges_km", "heldout_group"}
        if not isinstance(config[table], dict) or set(config[table]) != fields:
            raise ValueError(f"[{table}] fields must be exactly {sorted(fields)}")
    remote = config["remote"]
    if not isinstance(remote["host"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", remote["host"]):
        raise ValueError("remote.host must be an SSH alias or user@host")
    for key in ("repository", "workspace", "base_python", "uv"):
        value = remote[key]
        if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in PurePosixPath(value).parts:
            raise ValueError(f"remote.{key} must be a simple absolute POSIX path")
    repo, work = PurePosixPath(remote["repository"]), PurePosixPath(remote["workspace"])
    if work == PurePosixPath("/") or work == repo or work in repo.parents or repo in work.parents:
        raise ValueError("remote.workspace must be separate from the repository")
    v = config["validation"]
    if not isinstance(v["device"], str) or not re.fullmatch(r"cuda(?::[0-9]+)?", v["device"]):
        raise ValueError("validation.device must be cuda or cuda:N; CPU fallback is not permitted")
    for key in ("seed", "threads", "quick_timeout_s", "generation_timeout_s", "training_timeout_s"):
        positive(v[key], f"validation.{key}", integer=True, zero=key == "seed")
    if v["seed"] > 2**32 - 1:
        raise ValueError("validation.seed must be between 0 and 4294967295")
    d = config["dataset"]
    if d["event_name"] != "S5":
        raise ValueError("The current dataset generators support event_name=S5 only")
    for key in ("fs", "segment_duration_s", "segment_step_s", "half_duration_s", "test_step_s"):
        positive(d[key], f"dataset.{key}")
    for key in ("segment_start_s", "segment_end_s", "gap_s"):
        positive(d[key], f"dataset.{key}", zero=True)
    if d["segment_end_s"] < d["segment_start_s"]:
        raise ValueError("dataset.segment_end_s must be >= segment_start_s")
    for key in ("num_snapshots_per_segment", "snapshot_overlap_count"):
        positive(d[key], f"dataset.{key}", integer=True, zero=key == "snapshot_overlap_count")
    if d["snapshot_overlap_count"] >= d["num_snapshots_per_segment"]:
        raise ValueError("snapshot_overlap_count must be less than num_snapshots_per_segment")
    if d["num_snapshots_per_segment"] != 1 or d["snapshot_overlap_count"] != 0:
        raise ValueError("Remote comparison currently requires Ns=1 and no snapshot overlap; use separate generation configurations for Ns studies")
    if d["segment_step_s"] < d["segment_duration_s"]:
        raise ValueError("Remote comparison requires non-overlapping base windows")
    modes = d["frequency_selection_modes"]
    if not isinstance(modes, list) or not modes or any(not isinstance(m, str) or m not in {"full", "mel", "deep", "shallow", "adapt"} for m in modes):
        raise ValueError("Invalid dataset.frequency_selection_modes")
    if len(set(modes)) != len(modes) or ("full" in modes and len(modes) != 1):
        raise ValueError("Frequency modes must be unique; full must be used alone")
    if "adapt" in modes:
        raise ValueError("MoE_learn does not support adapt: frequency fitting must use only the retained optimizer-training subset")
    if d["split_strategy"] not in {"full_interleaved", "range_overlap"} or d["train_side"] not in {"before", "after"}:
        raise ValueError("Invalid split_strategy or train_side")
    ratio = d["train_test_ratio"]
    if not isinstance(ratio, list) or len(ratio) != 2:
        raise ValueError("train_test_ratio requires two positive integers")
    for value in ratio:
        positive(value, "dataset.train_test_ratio", integer=True)
    if not isinstance(d["manual_dataset_variant_tag"], str) or not re.fullmatch(r"[A-Za-z0-9_-]*", d["manual_dataset_variant_tag"]):
        raise ValueError("manual_dataset_variant_tag must contain letters, digits, underscores or hyphens")
    r = config["rbd"]
    if r["beam_selection"] not in {"best", "multipath"} or r["frequency_estimation"] not in {"selected", "full"}:
        raise ValueError("Invalid RBD beam selection or frequency estimation")
    if r["frequency_estimation"] == "full" and modes != ["full"]:
        raise ValueError("RBD full estimation requires shared frequency_selection_modes=[full]")
    if r["frequency_estimation"] == "selected" and "full" in modes:
        raise ValueError("RBD selected estimation cannot use full frequency selection")
    for key in ("use_plane_wave", "normalize_spectrum"):
        if not isinstance(r[key], bool):
            raise ValueError(f"rbd.{key} must be boolean")
    if d["stage_mode"] not in {"interleaved", "chronological", "before_then_after", "distance_holdout"}:
        raise ValueError("Invalid dataset.stage_mode")
    if d["stage_mode"] == "distance_holdout":
        if d["split_strategy"] != "range_overlap" or d["train_side"] != "before":
            raise ValueError("distance_holdout requires range_overlap and train_side=before")
        edges = d["distance_edges_km"]
        if not isinstance(edges, list) or len(edges) < 3:
            raise ValueError("distance_edges_km requires at least three edges")
        for edge in edges:
            positive(edge, "dataset.distance_edges_km")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("distance_edges_km must be strictly increasing")
        positive(d["heldout_group"], "dataset.heldout_group", integer=True)
        if d["heldout_group"] >= len(edges):
            raise ValueError("heldout_group must identify one distance interval")
        if config["continual"]["stages"] != len(edges) - 1:
            raise ValueError("distance_holdout stages must equal one initial stage plus the non-heldout groups")
    if d["stage_mode"] == "before_then_after":
        if d["split_strategy"] != "range_overlap" or d["train_side"] != "before":
            raise ValueError("before_then_after requires range_overlap and train_side=before")
        positive(d["after_test_fraction"], "dataset.after_test_fraction")
        if d["after_test_fraction"] >= 1:
            raise ValueError("dataset.after_test_fraction must be in (0, 1)")
        if d["after_grouping"] not in {"time_interleaved", "range_bins"}:
            raise ValueError("dataset.after_grouping must be time_interleaved or range_bins")
        if d["after_grouping"] == "range_bins":
            edges = d["after_range_edges_km"]
            if not isinstance(edges, list) or len(edges) != config["continual"]["stages"]:
                raise ValueError("after_range_edges_km length must equal continual.stages")
            for edge in edges:
                positive(edge, "dataset.after_range_edges_km")
            if any(left >= right for left, right in zip(edges, edges[1:])):
                raise ValueError("after_range_edges_km must be strictly increasing")
    positive(d["val_fraction"], "dataset.val_fraction")
    if d["val_fraction"] >= 1:
        raise ValueError("dataset.val_fraction must be in (0, 1)")
    t = config["training"]
    for key in ("initial_epochs", "adapt_epochs", "batch_size", "base_channels"):
        positive(t[key], f"training.{key}", integer=True)
    for key in ("warmup_epochs", "seed"):
        positive(t[key], f"training.{key}", integer=True, zero=True)
    if t["seed"] > 2**32 - 1 or t["seed"] != v["seed"]:
        raise ValueError("training.seed must equal validation.seed and be between 0 and 4294967295")
    if t["base_channels"] < 2:
        raise ValueError("training.base_channels must be at least 2")
    for key in ("learning_rate", "old_expert_learning_rate", "new_expert_learning_rate", "router_learning_rate"):
        positive(t[key], f"training.{key}")
    if t["old_expert_learning_rate"] >= t["new_expert_learning_rate"]:
        raise ValueError("Require 0 < old_expert_learning_rate < new_expert_learning_rate")
    positive(t["weight_decay"], "training.weight_decay", zero=True)
    continual = config["continual"]
    for key in ("stages", "initial_experts", "max_experts"):
        positive(continual[key], f"continual.{key}", integer=True)
    if continual["stages"] < 2:
        raise ValueError("continual.stages must be at least 2")
    if continual["initial_experts"] < 2 or continual["initial_experts"] > continual["max_experts"]:
        raise ValueError("Require 2 <= continual.initial_experts <= continual.max_experts")
    if continual["max_experts"] > 8:
        raise ValueError("continual.max_experts must not exceed 8")
    positive(continual["replay_size"], "continual.replay_size", integer=True, zero=True)
    if continual["replay_mode"] not in {"none", "buffer", "full"}:
        raise ValueError("continual.replay_mode must be none, buffer or full")
    if (continual["replay_mode"] == "buffer") != (continual["replay_size"] > 0):
        raise ValueError("replay_mode none/full requires replay_size=0; buffer requires positive replay_size")
    if type(continual["history_validation"]) is not bool:
        raise ValueError("continual.history_validation must be boolean")
    controller = config["controller"]
    for key in ("min_relative_gain", "max_old_relative_regression"):
        positive(controller[key], f"controller.{key}", zero=True)
        if controller[key] > 1:
            raise ValueError(f"controller.{key} must be in [0, 1]")
    positive(controller["novelty_threshold"], "controller.novelty_threshold")
    positive(controller["balance_loss_weight"], "controller.balance_loss_weight", zero=True)

    from scripts.MoE_learn.data_processing.generation_config import normalize_config
    normalize_config({"schema_version": 1, "dataset": {k: value for k, value in d.items()
                     if k not in {"stage_mode", "val_fraction", "after_test_fraction", "after_grouping", "after_range_edges_km", "distance_edges_km", "heldout_group"}}, "rbd": r})
    return config


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return validate_config(tomllib.load(handle))
