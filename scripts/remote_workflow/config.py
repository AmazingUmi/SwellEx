"""Strict configuration shared by local dispatch and remote execution."""
from __future__ import annotations

import copy
import math
from pathlib import Path, PurePosixPath
import re
import tomllib

from scripts.data_processing.generation_config import (
    FREQUENCY_KEYS,
    MULTIPATH_DEFAULTS,
    normalize_config,
    validate_generation_tables,
)
from scripts.data_processing.input_contract import METHODS
from scripts.network_training.catalog import ARCHITECTURES

FIELDS = {
    "experiment": {"methods", "networks"},
    "remote": {"host", "repository", "workspace", "base_python", "uv"},
    "validation": {
        "device",
        "seed",
        "threads",
        "quick_timeout_s",
        "generation_timeout_s",
        "training_timeout_s",
    },
    "dataset": {
        "event_name",
        "ctd_file",
        "fs",
        "segment_start_s",
        "segment_end_s",
        "segment_duration_s",
        "segment_step_s",
        "signal_centering",
        "fft_window",
        "frequency_selection_modes",
        "frequency_selection_config",
        "split_strategy",
        "train_test_ratio",
        "train_side",
        "half_duration_s",
        "gap_s",
        "test_step_s",
        "manual_dataset_variant_tag",
    },
    "rbd": {
        "beam_selection",
        "frequency_estimation",
        "use_plane_wave",
        "normalize_spectrum",
        "angle_min_deg",
        "angle_max_deg",
        "num_angles",
        "multipath_options",
    },
    "scm": {
        "num_snapshots_per_segment",
        "snapshot_overlap_count",
        "norm_floor",
    },
    "training": {
        "epochs",
        "batch_size",
        "predict_batch_size",
        "lr",
        "weight_decay",
        "base_channels",
        "dropout",
        "loss_space",
        "huber_beta",
        "normalize_input",
        "val_split_mode",
        "val_fraction",
        "val_gap_s",
        "num_workers",
    },
    "grnn": {
        "cv_folds",
        "spread",
        "sigma_candidates",
        "standardize_input",
        "per_sample_channel_norm",
        "batch_size",
        "predict_batch_size",
        "max_reference_samples",
        "max_validation_samples",
        "max_predict_samples",
    },
}


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

    exp = config["experiment"]
    methods = exp["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(m, str) or m not in METHODS for m in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError(f"experiment.methods must be a nonempty list of unique values from {METHODS}")
    networks = exp["networks"]
    if (
        not isinstance(networks, list)
        or not networks
        or any(not isinstance(n, str) or n not in ARCHITECTURES for n in networks)
        or len(set(networks)) != len(networks)
    ):
        raise ValueError(f"experiment.networks must be a nonempty list of unique values from {ARCHITECTURES}")

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
    if not isinstance(d["frequency_selection_modes"], list):
        raise ValueError("dataset.frequency_selection_modes must be a list")
    freq_cfg = d["frequency_selection_config"]
    if not isinstance(freq_cfg, dict) or set(freq_cfg) != set(FREQUENCY_KEYS):
        raise ValueError(f"[dataset.frequency_selection_config] fields must be exactly {sorted(FREQUENCY_KEYS)}")

    r = config["rbd"]
    mp_cfg = r["multipath_options"]
    if not isinstance(mp_cfg, dict) or set(mp_cfg) != set(MULTIPATH_DEFAULTS):
        raise ValueError(f"[rbd.multipath_options] fields must be exactly {sorted(MULTIPATH_DEFAULTS)}")

    s = config["scm"]
    validate_generation_tables(
        copy.deepcopy(d),
        copy.deepcopy(r),
        copy.deepcopy(s),
        methods=tuple(methods),
    )

    t = config["training"]
    for key in ("epochs", "batch_size", "predict_batch_size", "base_channels"):
        positive(t[key], f"training.{key}", integer=True)
    positive(t["num_workers"], "training.num_workers", integer=True, zero=True)
    positive(t["lr"], "training.lr")
    for key in ("weight_decay", "dropout", "huber_beta", "val_gap_s"):
        positive(t[key], f"training.{key}", zero=True)
    if t["dropout"] >= 1:
        raise ValueError("training.dropout must be < 1")
    positive(t["val_fraction"], "training.val_fraction")
    if t["val_fraction"] >= 1:
        raise ValueError("training.val_fraction must be < 1")
    if not isinstance(t["loss_space"], str) or t["loss_space"] not in {"km", "normalized"}:
        raise ValueError("Invalid training.loss_space")
    if not isinstance(t["normalize_input"], bool):
        raise ValueError("training.normalize_input must be boolean")
    if not isinstance(t["val_split_mode"], str) or t["val_split_mode"] not in {"random", "blocked"}:
        raise ValueError("Invalid training.val_split_mode")

    g = config["grnn"]
    positive(g["cv_folds"], "grnn.cv_folds", integer=True, zero=True)
    if g["cv_folds"] == 1:
        raise ValueError("grnn.cv_folds must be 0 or >= 2")
    positive(g["spread"], "grnn.spread")
    if not isinstance(g["sigma_candidates"], list):
        raise ValueError("grnn.sigma_candidates must be a list")
    if g["cv_folds"] >= 2 and not g["sigma_candidates"]:
        raise ValueError("grnn.sigma_candidates must be a nonempty list when cv_folds >= 2")
    for value in g["sigma_candidates"]:
        positive(value, "grnn.sigma_candidates")
    for key in ("standardize_input", "per_sample_channel_norm"):
        if not isinstance(g[key], bool):
            raise ValueError(f"grnn.{key} must be boolean")
    for key in ("batch_size", "predict_batch_size"):
        positive(g[key], f"grnn.{key}", integer=True)
    for key in ("max_reference_samples", "max_validation_samples", "max_predict_samples"):
        positive(g[key], f"grnn.{key}", integer=True, zero=True)
    return config


def generation_methods(config: dict, plan: list[dict]) -> list[str]:
    planned = [m for m in METHODS if any(job["method"] == m for job in plan)]
    return planned if planned else list(config["experiment"]["methods"])


def generation_payload(
    config: dict,
    result_path: str = "outputs/validation/generation_result.json",
    *,
    methods: list[str] | tuple[str, ...] | None = None,
) -> dict:
    effective_methods = config["experiment"]["methods"] if methods is None else methods
    return normalize_config(
        {
            "schema_version": 1,
            "dataset": copy.deepcopy(config["dataset"]),
            "rbd": copy.deepcopy(config["rbd"]),
            "scm": copy.deepcopy(config["scm"]),
            "result_path": result_path,
        },
        methods=effective_methods,
    )


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return validate_config(tomllib.load(handle))
