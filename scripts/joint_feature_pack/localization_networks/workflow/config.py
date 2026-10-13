"""Strict configuration shared by local dispatch and remote execution."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import re
import tomllib

FIELDS = {
    "remote": {"host", "repository", "workspace", "base_python", "uv"},
    "validation": {"device", "seed", "threads", "quick_timeout_s", "generation_timeout_s", "training_timeout_s"},
    "dataset": {"event_name", "fs", "segment_start_s", "segment_end_s", "segment_duration_s", "segment_step_s", "num_snapshots_per_segment", "snapshot_overlap_count", "frequency_selection_modes", "split_strategy", "train_test_ratio", "train_side", "half_duration_s", "gap_s", "test_step_s", "manual_dataset_variant_tag"},
    "rbd": {"beam_selection", "frequency_estimation", "use_plane_wave", "normalize_spectrum"},
    "training": {"epochs", "batch_size", "base_channels", "loss_space", "huber_beta", "val_fraction"},
    "fusion": {"mode"},
    "grnn": {"cv_folds", "sigma_candidates", "batch_size"},
}
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")


def positive(value: object, name: str, *, integer: bool = False, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != {"schema_version", *FIELDS}:
        raise ValueError("Configuration must contain schema_version and exactly the documented tables")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for table, fields in FIELDS.items():
        if not isinstance(config[table], dict) or set(config[table]) != fields:
            raise ValueError(f"[{table}] fields must be exactly {sorted(fields)}")
    if config["fusion"]["mode"] not in {"raw_concat", "block_rms"}:
        raise ValueError("fusion.mode must be raw_concat or block_rms")
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
    if not isinstance(v["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", v["device"]):
        raise ValueError("validation.device must be cpu, cuda, or cuda:N")
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
    t = config["training"]
    for key in ("epochs", "batch_size", "base_channels"):
        positive(t[key], f"training.{key}", integer=True)
    for key in ("huber_beta", "val_fraction"):
        positive(t[key], f"training.{key}")
    if t["val_fraction"] >= 1 or t["loss_space"] not in {"km", "normalized"}:
        raise ValueError("Invalid training.val_fraction or loss_space")
    g = config["grnn"]
    for key in ("cv_folds", "batch_size"):
        positive(g[key], f"grnn.{key}", integer=True)
    if g["cv_folds"] < 2:
        raise ValueError("grnn.cv_folds must be >= 2")
    if not isinstance(g["sigma_candidates"], list) or not g["sigma_candidates"]:
        raise ValueError("grnn.sigma_candidates must be a nonempty list")
    for value in g["sigma_candidates"]:
        positive(value, "grnn.sigma_candidates")
    return config


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return validate_config(tomllib.load(handle))
