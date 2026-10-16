"""Run validation inside a frozen remote project; report every terminal outcome."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.data_scale_pack.localization_networks.workflow.config import validate_config
from scripts.run_identity import validate_filters, validate_run_identity
from scripts.data_scale_pack.localization_networks.workflow.processes import run_process
from scripts.data_scale_pack.localization_networks.workflow.tasks import SUITES, entry_script, expected_phases, model_plan, package_name, resolve_selection

PACK = "data_scale_pack"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(project: Path, manifest: dict) -> None:
    inventory = manifest["files"]
    seen = set()
    for item in inventory:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) in seen:
            raise ValueError(f"Invalid/duplicate manifest path: {rel}")
        seen.add(str(rel))
        path = project / str(rel)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
            raise ValueError(f"Snapshot contains missing or external source: {rel}")
        if path.stat().st_size != item["size"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {rel}")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("Source manifest digest mismatch")
    actual = {p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file() and not {"output", "outputs", "Origindata", "__pycache__"}.intersection(p.relative_to(project).parts)}
    if actual != seen:
        raise ValueError(f"Unexpected source inventory: {sorted(actual ^ seen)}")


def prediction_metrics(csv_path: Path, expected: list[tuple[int, float, float]] | None = None) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No predictions: {csv_path}")
    errors = []
    indices = []
    previous_time = None
    if expected is not None and len(expected) != len(rows):
        raise ValueError(f"Prediction count does not match test dataset in {csv_path}")
    for position, row in enumerate(rows):
        pred, target, moment = (float(row[key]) for key in ("pred_range_km", "true_range_km", "window_center_s"))
        reported_error = float(row["abs_error_km"])
        if not all(math.isfinite(x) for x in (pred, target, moment, reported_error)):
            raise ValueError(f"Nonfinite predictions in {csv_path}")
        error = abs(pred - target)
        if reported_error < 0 or abs(error - reported_error) > 2.1e-6:
            raise ValueError(f"Prediction error column does not match values in {csv_path}")
        index = int(row["source_segment_idx"])
        if int(row["sample"]) != position or index < 1 or (indices and index <= indices[-1]):
            raise ValueError(f"Prediction sample/source indices are invalid in {csv_path}")
        if previous_time is not None and moment <= previous_time:
            raise ValueError(f"Prediction timestamps are not increasing in {csv_path}")
        previous_time = moment
        if expected is not None:
            expected_index, expected_time, expected_target = expected[position]
            if index != expected_index or abs(moment-expected_time) > 1.1e-6 or abs(target-expected_target) > 1.1e-6:
                raise ValueError(f"Prediction rows differ from the test dataset in {csv_path}")
        errors.append(error)
        indices.append(index)
    worst = max(range(len(rows)), key=errors.__getitem__)
    return {"samples": len(rows), "mae_km": sum(errors) / len(rows), "rmse_km": math.sqrt(sum(x*x for x in errors) / len(rows)), "max_error_km": errors[worst], "max_error_window_center_s": float(rows[worst]["window_center_s"]), "source_segment_indices": indices}



def expected_controlled_budget(training: dict, train_count: int) -> dict:
    batch_size = training["batch_size"]
    steps = train_count // batch_size
    if steps < 1:
        raise ValueError("Controlled training requires at least one complete batch")
    updates = training.get("optimizer_updates")
    epochs = training["epochs"]
    interval = training["validation_interval_updates"] if updates is not None else steps
    total = updates if updates is not None else epochs * steps
    if total < math.ceil(train_count / batch_size):
        raise ValueError("Controlled budget cannot present every retained source")
    return {"version": "controlled-budget-v1", "mode": "updates" if updates is not None else "epochs",
            "requested_epochs": epochs, "requested_optimizer_updates": updates,
            "total_optimizer_updates": total, "validation_interval_updates": interval,
            "validation_checks": total // interval, "batch_size": batch_size, "drop_last": True,
            "steps_per_data_pass": steps, "controlled_batches": True}


def training_history_metrics(history: list[dict], best: dict, last: dict, *, train_wall_s: float, batch_size: int,
                             training_config: dict | None = None) -> dict:
    """Derive comparison evidence only after matching history and both artifacts."""
    if not history or [row.get("epoch") for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Training history epochs must start at one and be contiguous")
    if not math.isfinite(train_wall_s) or train_wall_s <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Training wall time and batch size must be positive")
    controlled = bool(training_config and training_config.get("controlled_batches", False))
    if controlled:
        budget = expected_controlled_budget(training_config, len(last["train_indices"]))
        if any(checkpoint.get("budget_plan") != budget for checkpoint in (best, last)):
            raise ValueError("Checkpoint budget differs from frozen training configuration")
        if len(history) != budget["validation_checks"]:
            raise ValueError("Validation checkpoint count differs from controlled budget")
        expected_steps = budget["validation_interval_updates"]
    else:
        if any("budget_plan" in checkpoint for checkpoint in (best, last)):
            raise ValueError("Controlled artifact cannot be interpreted as a legacy epoch run")
        expected_steps = math.ceil(len(last["train_indices"]) / batch_size)
    cumulative_steps, elapsed = 0, 0.0
    for row in history:
        for key in ("train_loss", "val_loss", "train_mae_km", "train_rmse_km", "val_mae_km", "val_rmse_km", "epoch_elapsed_s"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Training history has invalid {key}")
        if row["epoch_elapsed_s"] <= 0:
            raise ValueError("Training history requires positive epoch_elapsed_s")
        if type(row.get("optimizer_steps_epoch")) is not int or row["optimizer_steps_epoch"] != expected_steps:
            raise ValueError("Training optimizer step count differs from batch geometry")
        cumulative_steps += expected_steps
        if type(row.get("optimizer_steps")) is not int or row["optimizer_steps"] != cumulative_steps:
            raise ValueError("Training cumulative optimizer steps are inconsistent")
        elapsed += row["epoch_elapsed_s"]
        if controlled:
            if row.get("record_unit") != "validation_checkpoint" or row.get("validation_index") != row["epoch"]:
                raise ValueError("Controlled history requires validation-checkpoint record identity")
            if row.get("sample_presentations_record") != expected_steps * batch_size or row.get("sample_presentations") != cumulative_steps * batch_size:
                raise ValueError("Controlled history sample presentation count differs from full batches")
            if (row.get("data_passes_completed") != cumulative_steps // budget["steps_per_data_pass"]
                    or row.get("updates_in_current_pass") != cumulative_steps % budget["steps_per_data_pass"]):
                raise ValueError("Controlled data-pass counters differ from update count")
            if type(row.get("unique_train_sources_seen")) is not int or not 1 <= row["unique_train_sources_seen"] <= len(last["train_indices"]):
                raise ValueError("Controlled history has invalid unique training-source count")
            for key in ("batch_sequence_sha256", "record_batch_sequence_sha256", "lr_sequence_sha256", "seen_source_identity_sha256"):
                if not isinstance(row.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", row[key]):
                    raise ValueError(f"Controlled history is missing {key}")
    if elapsed > train_wall_s + 0.01:
        raise ValueError("Epoch train/validation time exceeds the enclosing training phase")
    best_row = min(history, key=lambda row: row["val_rmse_km"])
    if best.get("epoch") != best_row["epoch"] or last.get("epoch") != len(history):
        raise ValueError("Best/last checkpoint epochs disagree with training history")
    if best.get("history") != history[:best_row["epoch"]] or last.get("history") != history:
        raise ValueError("Checkpoint history differs from exported training history")
    if best.get("metrics") != best_row or last.get("metrics") != history[-1]:
        raise ValueError("Checkpoint metrics differ from exported training history")
    for checkpoint in (best, last):
        if checkpoint.get("best_rmse_km") != best_row["val_rmse_km"]:
            raise ValueError("Checkpoint best RMSE differs from the history minimum")
    metrics = {
        "best_epoch": best_row["epoch"],
        "best_val_rmse_km": best_row["val_rmse_km"],
        "best_val_mae_km": best_row["val_mae_km"],
        "best_train_loss": best_row["train_loss"],
        "best_train_mae_km": best_row["train_mae_km"],
        "best_train_rmse_km": best_row["train_rmse_km"],
        "total_train_wall_s": train_wall_s,
        "sum_epoch_elapsed_s": elapsed,
        "optimizer_steps": cumulative_steps,
        "parameter_count": last["scaling"]["parameter_count"],
    }
    if controlled:
        if cumulative_steps != budget["total_optimizer_updates"]:
            raise ValueError("Final optimizer update count differs from controlled budget")
        if history[-1]["unique_train_sources_seen"] != len(last["train_indices"]):
            raise ValueError("Controlled budget did not expose every retained training source")
        for key in ("sampler_plan", "scheduler_plan", "initial_model_state_sha256"):
            if best.get(key) != last.get(key) or key not in last:
                raise ValueError(f"Controlled checkpoint metadata differs or is missing: {key}")
        if not isinstance(last["initial_model_state_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", last["initial_model_state_sha256"]):
            raise ValueError("Invalid initial-model state identity")
        pass_digests = last.get("training_progress", {}).get("data_pass_batch_sequence_sha256")
        if (not isinstance(pass_digests, list) or len(pass_digests) != history[-1]["data_passes_completed"]
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in pass_digests)):
            raise ValueError("Controlled checkpoint lacks complete data-pass batch-prefix hashes")
        if history[-1]["updates_in_current_pass"] == 0 and pass_digests[-1] != history[-1]["batch_sequence_sha256"]:
            raise ValueError("Final data-pass batch-prefix hash differs from the completed training stream")
        metrics.update(history_record_unit="validation_checkpoint", best_validation_index=best_row["validation_index"],
                       best_optimizer_step=best_row["optimizer_steps"], validation_checks=len(history),
                       budget_plan=budget, sampler_plan=last["sampler_plan"], scheduler_plan=last["scheduler_plan"],
                       initial_model_state_sha256=last["initial_model_state_sha256"],
                       final_batch_sequence_sha256=history[-1]["batch_sequence_sha256"],
                       final_lr_sequence_sha256=history[-1]["lr_sequence_sha256"],
                       final_seen_source_identity_sha256=history[-1]["seen_source_identity_sha256"],
                       data_pass_batch_sequence_sha256=list(pass_digests),
                       sample_presentations=history[-1]["sample_presentations"],
                       unique_train_sources_seen=history[-1]["unique_train_sources_seen"])
    return metrics


def workspace_environment(config: dict, lock_hash: str) -> Path:
    return Path(config["remote"]["workspace"]) / "environments" / lock_hash / ".venv"


def validate_controlled_comparison(metrics: list[dict]) -> None:
    """Different model initializations must not change the controlled data stream."""
    if not metrics:
        raise ValueError("Controlled comparison requires observations")
    keys = ("budget_plan", "sampler_plan", "scheduler_plan", "train_identity_sha256", "val_identity_sha256",
            "test_identity_sha256", "final_batch_sequence_sha256", "final_lr_sequence_sha256",
            "final_seen_source_identity_sha256", "data_pass_batch_sequence_sha256",
            "unique_train_sources_seen", "sample_presentations")
    first = metrics[0]
    for row in metrics:
        if any(key not in row or row[key] != first.get(key) for key in keys):
            raise ValueError("Controlled models differ in training membership, budget, batch stream, or learning-rate sequence")
        if row.get("scaling", {}).get("train_count") != first.get("scaling", {}).get("train_count"):
            raise ValueError("Controlled models have different retained training counts")


class Worker:
    def __init__(self, run: Path):
        self.run = run.resolve()
        self.project = self.run / "project"
        self.report = {"schema_version": 1, "pack": PACK, "run_id": self.run.name, "status": "RUNNING", "started_at": stamp(), "checks": [], "purpose": "engineering_validation"}
        self.config = None
        self.training_inventories = []
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}

    def save(self) -> None:
        self.report["updated_at"] = stamp()
        write_json(self.run / "report.json", self.report)

    def phase(self, name: str, action) -> object:
        row = {"name": name, "status": "RUNNING", "started_at": stamp()}
        self.report["checks"].append(row)
        self.save()
        started = time.monotonic()
        print(f"START {name}", flush=True)
        try:
            value = action()
            row.update(status="PASS", elapsed_s=round(time.monotonic()-started, 3))
            if isinstance(value, dict):
                row["result"] = value
            print(f"PASS {name}", flush=True)
            return value
        except BaseException as exc:
            row.update(status="FAIL", detail=str(exc), elapsed_s=round(time.monotonic()-started, 3))
            raise
        finally:
            row["finished_at"] = stamp()
            self.save()

    def command(self, name: str, argv: list[str], timeout: int, *, env: dict | None = None) -> None:
        log_path = self.run / "logs" / f"{name}.log"
        write_json(self.run / "logs" / f"{name}.command.json", {"argv": argv, "cwd": str(self.project), "timeout_s": timeout})
        with log_path.open("wb") as log:
            result = run_process(argv, cwd=self.project, env=self.env if env is None else env,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        if code:
            tail = log_path.read_text(errors="replace")[-2500:]
            raise RuntimeError(f"{name} exited {code}; see {log_path}\n{tail}")

    def json_command(self, name: str, argv: list[str], report_path: Path, timeout: int, *, env: dict | None = None) -> dict:
        self.command(name, argv, timeout, env=env)
        result = json.loads(report_path.read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"{name} did not produce a PASS report: {result}")
        return result

    def setup(self) -> dict:
        request = json.loads((self.run / "request.json").read_text())
        if not isinstance(request, dict):
            raise ValueError("Invalid remote request schema")
        fields = {"schema_version", "pack", "run_id", "suite", "config", "use_cache"}
        if request.get("suite") in {"selected", "budget"}:
            fields.add("selection")
        if set(request) != fields or type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Invalid remote request schema")
        if request["pack"] != PACK:
            raise ValueError("Remote request pack mismatch")
        if request["run_id"] != self.run.name or request["suite"] not in SUITES or not isinstance(request["use_cache"], bool):
            raise ValueError("Invalid run ID, suite or cache policy")
        if request["suite"] in {"selected", "budget"}:
            validate_filters(PACK, request["selection"])
        self.request = request
        self.config = validate_config(request["config"])
        validate_run_identity(self.run.name, PACK, self.config, request.get("selection"))
        expected = Path(self.config["remote"]["workspace"]) / "runs" / self.run.name
        if self.run != expected or self.project.resolve() != self.project:
            raise ValueError("Remote run is outside configured snapshot location")
        manifest = json.loads((self.run / "manifest.json").read_text())
        if manifest.get("pack") != PACK or manifest["run_id"] != self.run.name or manifest["suite"] != request["suite"]:
            raise ValueError("Manifest pack/run/suite mismatch")
        if request["suite"] in {"selected", "budget"}:
            selection = validate_filters(PACK, manifest.get("selection"))
            if selection != request["selection"]:
                raise ValueError("Manifest selection differs from the frozen request")
        elif "selection" in manifest:
            raise ValueError("Validation manifests cannot contain a selection")
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if config_hash != manifest["config_sha256"]:
            raise ValueError("Request configuration hash mismatch")
        verify_snapshot(self.project, manifest)
        if "calibration" in self.config:
            calibration = self.config["calibration"]
            artifact = self.project / calibration["path"]
            if not artifact.is_file() or artifact.is_symlink() or file_hash(artifact) != calibration["sha256"]:
                raise ValueError("Frozen calibration artifact is missing or differs from configuration")
            self.report["calibration"] = dict(calibration)
        generation = json.loads((self.run / "generation_config.json").read_text())
        expected_generation = {"schema_version": 1, "dataset": self.config["dataset"], "rbd": self.config["rbd"], "result_path": "outputs/validation/generation_result.json"}
        if generation != expected_generation:
            raise ValueError("Generation configuration differs from the frozen request")
        self.manifest = manifest
        self.report.update(suite=request["suite"], source_sha256=manifest["source_sha256"], git_head=manifest["git_head"], config_sha256=config_hash,
                           calibration_status="uncalibrated",
                           model_plan=model_plan(request["suite"], request.get("selection")))
        if request["suite"] == "experiment":
            self.report["purpose"] = "configured_experiment_without_accuracy_acceptance_threshold"
        elif request["suite"] in {"selected", "budget"}:
            self.report.update(selection=dict(request["selection"]), purpose="configured_model_selection_without_accuracy_acceptance_threshold")
        if request["suite"] == "budget":
            if not self.config["training"].get("controlled_batches", False) or "calibration" not in self.config:
                raise ValueError("Budget suite requires controlled batches and frozen calibration provenance")
            self.report["purpose"] = "controlled_epoch_vs_update_budget_with_frozen_validation_calibration"
        threads = str(self.config["validation"]["threads"])
        self.env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        self.python = sys.executable
        self.lock_hash = file_hash(self.project / "uv.lock")
        expected_environment = workspace_environment(self.config, self.lock_hash)
        if Path(sys.prefix).resolve() != expected_environment.resolve():
            raise ValueError("Worker must run inside the uv environment for the frozen lockfile")
        self.report["uv_lock_sha256"] = self.lock_hash
        return {"files": len(manifest["files"]), "source_sha256": manifest["source_sha256"]}

    def environment(self) -> dict:
        import torch
        info = {"python": sys.version, "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scipy", "h5py", "matplotlib")}, "cuda_available": torch.cuda.is_available()}
        device = torch.device(self.config["validation"]["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA validation requires an available GPU")
            info["gpu"] = torch.cuda.get_device_name(device)
        info.update(uv_lock_sha256=self.lock_hash, uv_environment=str(Path(sys.prefix)), uv_version=subprocess.check_output([self.config["remote"]["uv"], "--version"], text=True).strip(), operating_system=platform.freedesktop_os_release().get("PRETTY_NAME"), libc=platform.libc_ver())
        write_json(self.run / "environment.json", info)
        return info

    def data_identity(self) -> dict:
        raw = Path(self.config["remote"]["repository"]) / "Origindata"
        if (self.project / "Origindata").resolve() != raw.resolve():
            raise ValueError("Snapshot original-data link has an unexpected target")
        relative = ["positions/positions_vla.txt", "events/range/RangeEventS5/SproulToVLA.S5.txt", "environments/ctds/i9605.prn"]
        channels = sorted((raw / "events/S5/vla_matfiles").glob("S5_VLA_NO_*.mat"))
        if len(channels) != 21:
            raise ValueError(f"Expected 21 S5 VLA MAT files, found {len(channels)}")
        relative += [p.relative_to(raw).as_posix() for p in channels]
        inputs = [{"path": name, "size": (raw/name).stat().st_size, "sha256": file_hash(raw/name)} for name in relative]
        generation_sources = [item for item in self.manifest["files"] if item["path"].startswith("scripts/data_scale_pack/data_processing/") and item["path"].endswith(".py")]
        shared_validators = {"scripts/data_scale_pack/localization_networks/common/h5_utils.py",
                             "scripts/data_scale_pack/localization_networks/common/input_contract.py"}
        validator = [item for item in self.manifest["files"]
                     if item["path"].endswith(".py") and "tests" not in PurePosixPath(item["path"]).parts
                     and (item["path"].startswith("scripts/data_scale_pack/localization_networks/validation/")
                          or item["path"] in shared_validators)]
        cache_identity = {"schema_version": 1, "pack": PACK, "dataset": self.config["dataset"], "rbd": self.config["rbd"], "inputs": inputs, "generator_sources": generation_sources, "validator_sources": validator, "uv_lock_sha256": self.lock_hash, "numerical_dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")}}
        key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        write_json(self.run / "data_identity.json", {"cache_key": key, **cache_identity})
        self.cache_key = key
        self.cache = Path(self.config["remote"]["workspace"]) / "datasets" / key
        return {"inputs": len(inputs), "cache_key": key}

    def datasets(self) -> dict:
        root = self.project / "outputs/Datasets"
        root.parent.mkdir(parents=True, exist_ok=True)
        generation_report = self.project / "outputs/validation/generation_result.json"
        cached = self.request["use_cache"] and (self.cache / "provenance.json").is_file()
        if cached:
            provenance = json.loads((self.cache / "provenance.json").read_text())
            if provenance.get("pack") != PACK or provenance["cache_key"] != self.cache_key:
                raise ValueError("Dataset cache identity mismatch")
            cached_root = self.cache / "Datasets"
            if cached_root.resolve() != cached_root or not cached_root.is_dir():
                raise ValueError("Invalid cached dataset directory")
            current = {p.relative_to(cached_root).as_posix(): file_hash(p) for p in cached_root.rglob("*") if p.is_file()}
            if current != provenance["files"]:
                raise ValueError("Dataset cache content hash mismatch; use --no-cache to regenerate")
            root.symlink_to(cached_root, target_is_directory=True)
            dirs = {method: str(root / name) for method, name in provenance["dataset_names"].items()}
            self.report["dataset_cache"] = {"reused": True, "producer_run_id": provenance["producer_run_id"], "cache_key": self.cache_key}
            self.report["checks"].append({"name": "data_generation", "status": "SKIP", "detail": f"Reusing hash-verified datasets from {provenance['producer_run_id']}"})
        else:
            self.phase("data_generation", lambda: self.json_command("data_generation", [self.python, "-B", str(self.project / "scripts/data_scale_pack/data_processing/generate.py"), "--config", str(self.run / "generation_config.json")], generation_report, self.config["validation"]["generation_timeout_s"]))
            generated = json.loads(generation_report.read_text())
            dirs = generated["datasets"]
            self.report["dataset_cache"] = {"reused": False, "producer_run_id": self.run.name, "cache_key": self.cache_key}
        if set(dirs) != {"RBD", "ELM", "SCM"}:
            raise ValueError("Generation must produce RBD, ELM and SCM directories")
        for value in dirs.values():
            if not Path(value).resolve().is_relative_to(root.resolve()):
                raise ValueError("Generated dataset path is outside dataset root")
        self.dataset_dirs = dirs
        report_path = self.run / "reports/datasets.json"
        result = self.phase("dataset_contract", lambda: self.json_command("dataset_contract", [self.python, "-B", str(self.project / "scripts/data_scale_pack/localization_networks/validation/validate.py"), "datasets", "--root", str(root), "--report", str(report_path), "--expected-config", str(self.run / "generation_config.json")], report_path, self.config["validation"]["quick_timeout_s"]))
        if not cached and self.request["use_cache"] and not self.cache.exists():
            self.cache.mkdir(parents=True)
            shutil.move(str(root), str(self.cache / "Datasets"))
            root.symlink_to(self.cache / "Datasets", target_is_directory=True)
            provenance = {"pack": PACK, "cache_key": self.cache_key, "producer_run_id": self.run.name, "source_sha256": self.manifest["source_sha256"], "dataset_names": {m: Path(p).relative_to(root).as_posix() for m,p in dirs.items()}, "files": {p.relative_to(self.cache / "Datasets").as_posix(): file_hash(p) for p in (self.cache / "Datasets").rglob("*") if p.is_file()}}
            write_json(self.cache / "provenance.json", provenance)
        self.report["dataset_directories"] = dirs
        self.save()
        return result

    def scaling_arguments(self) -> list[str]:
        scaling = self.config["scaling"]
        return ["--train-fraction", str(self.config["training"]["train_fraction"]),
                "--scale-reference-samples", str(scaling["reference_samples"]),
                "--scale-exponent", str(scaling["exponent"])]

    def budget_arguments(self) -> list[str]:
        training = self.config["training"]
        updates = training.get("optimizer_updates")
        arguments = (["--epochs", str(training["epochs"])] if updates is None else
                     ["--optimizer-updates", str(updates), "--validation-interval-updates", str(training["validation_interval_updates"])])
        if training.get("controlled_batches", False):
            arguments.append("--controlled-batches")
        return arguments

    def resolved_architecture(self, method: str, model: str, checkpoint_path: Path) -> dict:
        """Check the exported architecture against the artifact used for prediction."""
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        paths = list(checkpoint_path.parent.glob("*_architecture_*.json"))
        if len(paths) != 1:
            raise ValueError("Expected one frozen architecture sidecar")
        sidecar = json.loads(paths[0].read_text())
        if any(sidecar.get(key) != value for key, value in (("pack", PACK), ("method", method), ("model", model))):
            raise ValueError("Architecture sidecar pack/method/model mismatch")
        if checkpoint.get("pack") != PACK or checkpoint.get("method") != method or checkpoint.get("model_name") != model:
            raise ValueError("Checkpoint pack/method/model mismatch")
        if checkpoint.get("train_fraction") != self.config["training"]["train_fraction"]:
            raise ValueError("Checkpoint pack or training fraction differs from the frozen request")
        scaling = checkpoint["scaling"]
        if (sidecar.get("scaling") != scaling or sidecar.get("model_config") != checkpoint["model_config"]
                or sidecar.get("sampling") != checkpoint["sampling"]
                or sidecar.get("training_inventory") != checkpoint["training_inventory"]):
            raise ValueError("Architecture sidecar differs from checkpoint")
        if scaling["scaling_policy"] != self.config["scaling"] or scaling["train_count"] != len(checkpoint["train_indices"]):
            raise ValueError("Resolved scaling differs from policy or actual training count")
        if self.config["training"].get("controlled_batches", False):
            for key in ("budget_plan", "sampler_plan", "scheduler_plan", "initial_model_state_sha256"):
                if key not in checkpoint or sidecar.get(key) != checkpoint[key]:
                    raise ValueError(f"Controlled architecture sidecar differs from checkpoint: {key}")
        return {"scaling": scaling, "architecture_path": paths[0].relative_to(self.run).as_posix()}

    def training_evidence(self, method: str, model: str, best_path: Path, history: list[dict], test_rows: list[tuple[int, float, float]]) -> dict:
        """Reconstruct physical split membership from immutable HDF5 inputs."""
        import importlib
        import torch
        from scripts.data_scale_pack.localization_networks.common.checkpoints import validate_checkpoint
        from scripts.data_scale_pack.localization_networks.common.data_selection import source_identity
        from scripts.data_scale_pack.localization_networks.common.training import dataset_split_inventory, range_coverage, split_inventory

        last_paths = list(best_path.parent.glob("*_last_*.pt"))
        if len(last_paths) != 1:
            raise ValueError("Expected one last checkpoint for training evidence")
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(last_paths[0], map_location="cpu", weights_only=True)
        for checkpoint in (best, last):
            validate_checkpoint(checkpoint, method=method, model_name=model, dataset_name=self.dataset_dirs[method])
        for key in ("model_config", "scaling", "sampling", "source_identity", "train_indices", "val_indices", "training_inventory", "split_mode", "seed"):
            if best[key] != last[key]:
                raise ValueError(f"Best/last checkpoint training evidence differs: {key}")
        if Path(last["best_checkpoint_path"]).resolve() != best_path.resolve():
            raise ValueError("Last checkpoint names a different best artifact")
        if last["seed"] != self.config["validation"]["seed"]:
            raise ValueError("Checkpoint seed differs from frozen request")
        paths = last["source_identity"]
        train_paths = [Path(item["path"]) for item in paths["train_sources"]]
        val_paths = [Path(item["path"]) for item in paths["validation_sources"]]
        if source_identity(train_paths, val_paths) != paths:
            raise ValueError("Training input bytes differ from checkpoint source identity")
        bindings = importlib.import_module(f"scripts.data_scale_pack.localization_networks.{method.lower()}.network.bindings").BINDINGS
        train_dataset = bindings.dataset_class(train_paths, normalize_input=False)
        val_dataset = train_dataset if last["split_mode"].endswith("random_split") else bindings.dataset_class(val_paths, normalize_input=False)
        try:
            train = dataset_split_inventory(train_dataset, last["train_indices"])
            validation = dataset_split_inventory(val_dataset, last["val_indices"])
        finally:
            train_dataset.close()
            if val_dataset is not train_dataset:
                val_dataset.close()
        inventory = {"schema_version": 1, "train": train, "validation": validation,
                     "validation_coverage": range_coverage(train, validation)}
        if inventory != last["training_inventory"]:
            raise ValueError("Training membership inventory differs from original HDF5 rows")
        if self.config["training"].get("controlled_batches", False):
            if last["training_progress"]["seen_source_segment_indices"] != sorted(train["source_segment_indices"]):
                raise ValueError("Controlled training did not expose every retained HDF5 source")
        if set(train["source_segment_indices"]) & set(validation["source_segment_indices"]):
            raise ValueError("Training and validation inventories overlap")
        test = split_inventory(test_rows)
        if (set(train["source_segment_indices"]) | set(validation["source_segment_indices"])) & set(test["source_segment_indices"]):
            raise ValueError("Training/validation and held-out test inventories overlap")
        phases = [row for row in self.report["checks"] if row["name"] == f"train_{method}_{model}" and row["status"] == "PASS"]
        if len(phases) != 1:
            raise ValueError("Expected one completed training phase for wall-time evidence")
        metrics = training_history_metrics(history, best, last, train_wall_s=phases[0]["elapsed_s"], batch_size=self.config["training"]["batch_size"],
                                           training_config=self.config["training"])
        metrics.update(train_identity_sha256=train["identity_sha256"], val_identity_sha256=validation["identity_sha256"], test_identity_sha256=test["identity_sha256"])
        self.training_inventories.append({
            "method": method, "model": model, "seed": last["seed"], "train_fraction": last["train_fraction"],
            "train_indices": last["train_indices"], "val_indices": last["val_indices"],
            "source_identity": paths, "sampling": last["sampling"], "training_inventory": inventory,
            "test": test, "test_coverage": range_coverage(train, test), "training_metrics": metrics,
        })
        inventory_path = self.run / "reports/training_inventory.json"
        write_json(inventory_path, {"schema_version": 1, "pack": PACK, "run_id": self.run.name, "models": self.training_inventories})
        return {**metrics, "training_inventory_path": inventory_path.relative_to(self.run).as_posix()}

    def train_predict(self, method: str, model: str) -> dict:
        cfg, v = self.config["training"], self.config["validation"]
        data = self.dataset_dirs[method]
        output = self.project / "outputs/networks_results" / method
        entry = str(self.project / f"scripts/data_scale_pack/localization_networks/{package_name(method)}/main.py")
        name = f"{method}_{model}"
        train_args = [self.python, "-B", entry, "train", "--model", model, "--data", data, "--output-dir", str(output), "--no-resume", *self.budget_arguments(), "--batch-size", str(cfg["batch_size"]), "--base-channels", str(cfg["base_channels"]), "--loss-space", cfg["loss_space"], "--huber-beta", str(cfg["huber_beta"]), "--val-fraction", str(cfg["val_fraction"]), "--seed", str(v["seed"]), "--device", v["device"], "--num-workers", "0"]
        train_args.extend(self.scaling_arguments())
        self.phase(f"train_{name}", lambda: self.command(f"train_{name}", train_args, v["training_timeout_s"]))
        checkpoints = sorted((output / model / "train_outputs").glob("*_best_*.pt"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected one best checkpoint for {name}, got {checkpoints}")
        histories = list((output / model / "train_outputs").glob("*_history_*.json"))
        if len(histories) != 1:
            raise ValueError(f"Expected one training history for {name}")
        history = json.loads(histories[0].read_text())
        record_count = cfg["epochs"] if cfg.get("optimizer_updates") is None else cfg["optimizer_updates"] // cfg["validation_interval_updates"]
        if len(history) != record_count or [row["epoch"] for row in history] != list(range(1, record_count+1)):
            raise ValueError(f"Unexpected training epoch history for {name}")
        for row in history:
            for key in ("train_loss", "val_loss", "train_mae_km", "val_mae_km", "val_rmse_km"):
                if not math.isfinite(row[key]):
                    raise ValueError(f"Nonfinite training metric {key} for {name}")
        self.phase(f"predict_{name}", lambda: self.command(f"predict_{name}", [self.python, "-B", entry, "predict", "--model", model, "--data", data, "--output-dir", str(output), "--checkpoint", str(checkpoints[0]), "--batch-size", str(cfg["batch_size"]), "--device", v["device"], "--print-first", "0", "--no-plot"], v["training_timeout_s"]))
        csvs = list((output / model / "test_outputs").glob("*_predictions_*.csv"))
        if len(csvs) != 1:
            raise ValueError(f"Expected one prediction CSV for {name}")
        test_rows = self.test_rows(data)
        metrics = prediction_metrics(csvs[0], test_rows)
        architecture = self.resolved_architecture(method, model, checkpoints[0])
        evidence = self.training_evidence(method, model, checkpoints[0], history, test_rows)
        return {"method": method, "model": model, "epochs": cfg["epochs"], "checkpoint": str(checkpoints[0]),
                "train_fraction": cfg["train_fraction"], **architecture, **metrics, **evidence}

    def test_rows(self, dataset_dir: str) -> list[tuple[int, float, float]]:
        import h5py
        import numpy as np
        rows = []
        for path in sorted(Path(dataset_dir).glob("*_test.h5")):
            with h5py.File(path, "r") as h5:
                labels = np.asarray(h5["/y_range_km"]).reshape(-1).astype(np.float32)
                valid = np.asarray(h5["/valid_sample"]).reshape(-1).astype(bool) & np.isfinite(labels)
                indices = np.asarray(h5["/split/source_segment_idx"]).reshape(-1)
                moments = np.asarray(h5["/time/window_center_s"]).reshape(-1)
                rows.extend((int(indices[i]), float(moments[i]), float(labels[i])) for i in np.flatnonzero(valid))
        if not rows:
            raise ValueError("Test HDF5 contains no valid rows")
        return rows

    def prediction_plot(self, directory: Path) -> None:
        plots = list(directory.glob("*.pdf"))
        if len(plots) != 1 or plots[0].is_symlink():
            raise ValueError(f"Expected one prediction PDF in {directory}")
        content = plots[0].read_bytes()
        if len(content) < 100 or not content.startswith(b"%PDF-") or not content.rstrip().endswith(b"%%EOF"):
            raise ValueError(f"Expected a complete prediction PDF in {directory}")

    def checkpoint_resume(self) -> None:
        # Isolated second output verifies actual optimizer/scheduler/checkpoint restore.
        cfg, v = self.config["training"], self.config["validation"]
        source = self.project / "outputs/networks_results/RBD/complex_cnn_range/train_outputs"
        last = list(source.glob("*_last_*.pt"))
        if len(last) != 1:
            raise ValueError("Expected one last checkpoint for resume validation")
        target = self.project / "outputs/resume_validation/RBD"
        self.command("checkpoint_resume", [self.python, "-B", str(self.project / entry_script(resolve_selection("RBD", "complex_CNN"))), "train", "--model", "complex_cnn_range", "--data", self.dataset_dirs["RBD"], "--output-dir", str(target), "--resume-checkpoint", str(last[0]), "--epochs", "1", "--batch-size", str(cfg["batch_size"]), "--base-channels", str(cfg["base_channels"]), "--seed", str(v["seed"]), "--loss-space", cfg["loss_space"], "--huber-beta", str(cfg["huber_beta"]), "--val-fraction", str(cfg["val_fraction"]), "--device", v["device"], *self.scaling_arguments()], v["training_timeout_s"])
        import torch
        resumed_paths = list(target.rglob("*_last_*.pt"))
        if len(resumed_paths) != 1:
            raise ValueError("Resume validation did not produce one checkpoint")
        resumed = torch.load(resumed_paths[0], map_location="cpu", weights_only=True)
        original = torch.load(last[0], map_location="cpu", weights_only=True)
        if resumed["epoch"] != original["epoch"] + 1 or resumed["train_indices"] != original["train_indices"] or resumed["val_indices"] != original["val_indices"]:
            raise ValueError("Resumed epoch/split differs from expected checkpoint")
        if resumed["scaling"] != original["scaling"] or resumed["model_config"] != original["model_config"]:
            raise ValueError("Resume recomputed or changed the frozen architecture")

    def export(self) -> None:
        target = self.run / "export" / "data"
        figures = self.run / "export" / "figure"
        target.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        for name in ("report.json", "manifest.json", "request.json", "generation_config.json", "environment.json", "data_identity.json"):
            if (self.run / name).is_file():
                shutil.copyfile(self.run / name, target / name)
        for directory in (self.run / "logs", self.run / "reports", self.project / "outputs/validation", self.project / "outputs/networks_results", self.project / "outputs/resume_validation"):
            if not directory.exists():
                continue
            prefix = directory.relative_to(self.run)
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".json", ".log", ".csv"}:
                    dest = target / prefix / path.relative_to(directory)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, dest)
        # Automatic figures have one authority: the renderer's verified CSV/
        # history tree. Raw manual prediction plots are not duplicated here.
        generated = self.project / "output" / "figure"
        if generated.exists():
            for path in generated.rglob("*"):
                if not path.is_file() or path.is_symlink():
                    continue
                suffix = path.suffix.lower()
                if suffix == ".pdf":
                    dest = figures / path.relative_to(generated)
                elif suffix in {".json", ".csv", ".log"}:
                    dest = target / "reports" / "visualization" / path.relative_to(generated)
                else:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)

    def visualize(self) -> dict:
        output = self.project / "output" / "figure"
        self.command("visualization", [self.python, "-B", str(self.project / "scripts/data_scale_pack/localization_networks/workflow/render.py"),
                                      "--run-dir", str(self.run), "--output-dir", str(output)],
                     self.config["validation"]["quick_timeout_s"])
        result = json.loads((output / "manifest.json").read_text())
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise ValueError("Visualization did not return a PASS manifest")
        for item in model_plan(self.request["suite"], self.request.get("selection")):
            self.prediction_plot(output / "models" / item["method"] / item["model"] / "test_outputs")
        write_json(self.run / "reports" / "visualization.json", result)
        return result

    def execute(self) -> int:
        self.save()
        lock = None
        code = 1
        planned_phases = {"snapshot_integrity"}
        try:
            self.phase("snapshot_integrity", self.setup)
            suite = self.request["suite"]
            selection = self.request.get("selection")
            planned_phases = expected_phases(suite, selection)
            workspace = Path(self.config["remote"]["workspace"])
            lock = (workspace / "validation.lock").open("a")
            self.report["status"] = "WAITING_FOR_RESOURCE"
            self.save()
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.report["status"] = "RUNNING"
            self.save()
            self.phase("environment", self.environment)
            v = self.config["validation"]
            if suite not in {"selected", "budget"}:
                python_report = self.run / "reports/python_quick.json"
                self.phase("python_quick", lambda: self.json_command("python_quick", [self.python, "-B", str(self.project / "scripts/data_scale_pack/localization_networks/validation/validate.py"), "quick", "--device", v["device"], "--report", str(python_report)], python_report, v["quick_timeout_s"]))
                numerical_report = self.project / "outputs/validation/data_numerical.json"
                self.phase("data_numerical", lambda: self.json_command("data_numerical", [self.python, "-B", str(self.project / "scripts/data_scale_pack/data_processing/validate.py"), "--report", str(numerical_report)], numerical_report, v["quick_timeout_s"]))
            if suite != "quick":
                self.phase("original_data_identity", self.data_identity)
                self.datasets()
                metrics = []
                for job in model_plan(suite, selection):
                    method, model = job["method"], job["model"]
                    metrics.append(self.phase(f"evaluate_{method}_{model}", lambda m=method,n=model: self.train_predict(m,n)))
                expected_indices = metrics[0]["source_segment_indices"]
                if any(row["source_segment_indices"] != expected_indices for row in metrics):
                    raise ValueError("Models did not evaluate the same test source windows")
                if self.config["training"].get("controlled_batches", False):
                    validate_controlled_comparison(metrics)
                if suite in {"smoke", "experiment"}:
                    self.phase("checkpoint_resume", self.checkpoint_resume)
                self.report["metrics"] = metrics
                self.save()
                write_json(self.run / "reports/metrics.json", {"status": "PASS", "purpose": self.report["purpose"], "metrics": metrics})
                self.phase("visualization", self.visualize)
            self.phase("source_unchanged_after_validation", lambda: verify_snapshot(self.project, self.manifest))
            self.report.update(status="PASS", exit_code=0)
            code = 0
        except BaseException as exc:
            self.report.update(status="FAIL", exit_code=1, error=f"{type(exc).__name__}: {exc}")
            print(self.report["error"], flush=True)
        finally:
            completed = {row["name"] for row in self.report["checks"]}
            self.report["not_run"] = sorted(planned_phases-completed)
            self.report["finished_at"] = stamp()
            terminal = self.report["status"]
            self.report["status"] = "EXPORTING"
            self.save()
            try:
                self.export()
                self.report["status"] = terminal
                self.save()
                write_json(self.run / "export/data/report.json", self.report)
            except Exception as exc:
                self.report.update(status="FAIL", exit_code=1, error=f"Result export failed: {exc}")
                self.save()
                code = 1
            if lock is not None:
                lock.close()
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(f"Remote worker received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    return Worker(args.run_dir).execute()


if __name__ == "__main__":
    raise SystemExit(main())
