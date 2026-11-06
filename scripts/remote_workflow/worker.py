"""Run validation inside a frozen remote project; report every terminal outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.remote_workflow.config import (
    generation_methods,
    generation_payload,
    validate_config,
)
from scripts.remote_workflow.metrics_contract import (
    compute_prediction_metrics,
    parse_prediction_file,
)
from scripts.remote_workflow.model_plan import (
    METHODS,
    SUITES,
    entry_script,
    expected_phases,
    model_plan,
    resolve_selection,
)
from scripts.remote_workflow.packs import DEFAULT_PACK
from scripts.remote_workflow.processes import run_process
from scripts.remote_workflow.run_identity import validate_filters, validate_run_identity

PACK = DEFAULT_PACK


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
    return compute_prediction_metrics(parse_prediction_file(csv_path, expected=expected))


def _count_vla_elements(positions_path: Path) -> int:
    if not positions_path.is_file():
        raise ValueError(f"Missing VLA positions file: {positions_path}")
    rows = 0
    for raw_line in positions_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"VLA positions row requires at least two columns: {positions_path}")
        try:
            values = (float(parts[0]), float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric VLA position in {positions_path}") from exc
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite VLA position value in {positions_path}")
        rows += 1
    if rows < 2:
        raise ValueError(f"VLA positions need at least two finite channel/depth rows: {positions_path}")
    return rows


def _link_cached_files(cached_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=False)
    for item in sorted(cached_dir.rglob("*")):
        if item.is_file():
            dest = target_dir / item.relative_to(cached_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(item)


def workspace_environment(config: dict, lock_hash: str) -> Path:
    return Path(config["remote"]["workspace"]) / "environments" / lock_hash / ".venv"


class Worker:
    def __init__(self, run: Path):
        self.run = run.resolve()
        self.project = self.run / "project"
        self.report = {"schema_version": 1, "pack": PACK, "run_id": self.run.name, "status": "RUNNING", "started_at": stamp(), "checks": [], "purpose": "engineering_validation"}
        self.request = None
        self.config = None
        self.required_methods = list(METHODS)
        self.cache_key = None
        self.method_cache_keys: dict[str, str] = {}
        self.method_caches: dict[str, Path] = {}
        self.dataset_dirs: dict[str, str] = {}
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
            row.update(status="PASS", elapsed_s=round(time.monotonic() - started, 3))
            if isinstance(value, dict):
                row["result"] = value
            print(f"PASS {name}", flush=True)
            return value
        except BaseException as exc:
            row.update(status="FAIL", detail=str(exc), elapsed_s=round(time.monotonic() - started, 3))
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
        if request.get("suite") == "selected":
            fields.add("selection")
        if set(request) != fields or type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Invalid remote request schema")
        if request["pack"] != PACK:
            raise ValueError("Remote request pack mismatch")
        if request["run_id"] != self.run.name or request["suite"] not in SUITES or not isinstance(request["use_cache"], bool):
            raise ValueError("Invalid run ID, suite or cache policy")
        if request["suite"] == "selected":
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
        if request["suite"] == "selected":
            selection = validate_filters(PACK, manifest.get("selection"))
            if selection != request["selection"]:
                raise ValueError("Manifest selection differs from the frozen request")
        elif "selection" in manifest:
            raise ValueError("Validation manifests cannot contain a selection")
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if config_hash != manifest["config_sha256"]:
            raise ValueError("Request configuration hash mismatch")
        verify_snapshot(self.project, manifest)
        plan = model_plan(request["suite"], request.get("selection"))
        self.required_methods = [m for m in METHODS if any(job["method"] == m for job in plan)]
        generation = json.loads((self.run / "generation_config.json").read_text())
        expected_generation = generation_payload(self.config, methods=generation_methods(self.config, plan))
        if generation != expected_generation:
            raise ValueError("Generation configuration differs from the frozen request")
        self.manifest = manifest
        self.report.update(suite=request["suite"], source_sha256=manifest["source_sha256"], git_head=manifest["git_head"], config_sha256=config_hash,
                           model_plan=plan)
        if request["suite"] == "experiment":
            self.report["purpose"] = "configured_experiment_without_accuracy_acceptance_threshold"
        elif request["suite"] == "selected":
            self.report.update(selection=validate_filters(PACK, request["selection"]), purpose="configured_model_selection_without_accuracy_acceptance_threshold")
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
        if self.request is not None:
            plan = model_plan(self.request["suite"], self.request.get("selection"))
            self.required_methods = [m for m in METHODS if any(job["method"] == m for job in plan)] or list(METHODS)
        else:
            self.required_methods = list(METHODS)
        num_elements = _count_vla_elements(raw / "positions/positions_vla.txt")
        event = self.config["dataset"]["event_name"]
        channels = sorted((raw / f"events/{event}/vla_matfiles").glob(f"{event}_VLA_NO_*.mat"))
        if len(channels) != num_elements:
            raise ValueError(f"Expected {num_elements} {event} VLA MAT files, found {len(channels)}")
        base_relative = [
            "positions/positions_vla.txt",
            f"events/range/RangeEvent{event}/SproulToVLA.{event}.txt",
            *(p.relative_to(raw).as_posix() for p in channels),
        ]
        resolved_raw = raw.resolve()
        for name in base_relative:
            candidate = raw / name
            if not candidate.is_file() or not candidate.resolve().is_relative_to(resolved_raw):
                raise ValueError(f"Missing or external raw input: {name}")
        base_inputs = [{"path": name, "size": (raw / name).stat().st_size, "sha256": file_hash(raw / name)} for name in base_relative]
        if "RBD" in self.required_methods:
            ctd_rel = f"environments/ctds/{self.config['dataset']['ctd_file']}"
            ctd_path = raw / ctd_rel
            if not ctd_path.is_file() or not ctd_path.resolve().is_relative_to(resolved_raw):
                raise ValueError(f"Missing or external CTD profile: {ctd_rel}")
            ctd_input = {"path": ctd_rel, "size": ctd_path.stat().st_size, "sha256": file_hash(ctd_path)}
            rbd_inputs = [base_inputs[0], base_inputs[1], ctd_input, *base_inputs[2:]]
            all_inputs = rbd_inputs
        else:
            rbd_inputs = []
            all_inputs = base_inputs

        generation_sources = [
            item for item in self.manifest["files"]
            if item["path"].endswith(".py")
            and "tests" not in PurePosixPath(item["path"]).parts
            and item["path"].startswith("scripts/data_processing/")
        ]
        shared_validators = {"scripts/network_training/common/h5_utils.py"}
        validator = [
            item for item in self.manifest["files"]
            if item["path"].endswith(".py")
            and "tests" not in PurePosixPath(item["path"]).parts
            and (item["path"].startswith("scripts/remote_workflow/validation/") or item["path"] in shared_validators)
        ]
        normalized = generation_payload(self.config, methods=self.required_methods)
        num_deps = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")}

        method_identities: dict[str, dict] = {}
        method_keys: dict[str, str] = {}
        for method in self.required_methods:
            if method == "RBD":
                m_dataset = dict(normalized["dataset"])
                m_estimator = {"rbd": normalized["rbd"]}
                m_inputs = rbd_inputs
            else:
                m_dataset = {k: v for k, v in normalized["dataset"].items() if k != "ctd_file"}
                m_estimator = {"scm": normalized["scm"]}
                m_inputs = base_inputs
            m_identity = {
                "schema_version": 1,
                "pack": PACK,
                "method": method,
                "dataset": m_dataset,
                **m_estimator,
                "inputs": m_inputs,
                "generator_sources": generation_sources,
                "validator_sources": validator,
                "uv_lock_sha256": self.lock_hash,
                "numerical_dependencies": num_deps,
            }
            method_identities[method] = m_identity
            method_keys[method] = hashlib.sha256(json.dumps(m_identity, sort_keys=True).encode()).hexdigest()

        overall = {
            "schema_version": 1,
            "pack": PACK,
            "methods": self.required_methods,
            "method_cache_keys": method_keys,
            "method_identities": method_identities,
        }
        key = hashlib.sha256(json.dumps(overall, sort_keys=True).encode()).hexdigest()
        write_json(
            self.run / "data_identity.json",
            {
                "schema_version": 1,
                "pack": PACK,
                "cache_key": key,
                "methods": self.required_methods,
                "method_cache_keys": method_keys,
                "inputs": all_inputs,
                "method_identities": method_identities,
                "generator_sources": generation_sources,
                "validator_sources": validator,
                "uv_lock_sha256": self.lock_hash,
                "numerical_dependencies": num_deps,
            },
        )
        workspace_datasets = Path(self.config["remote"]["workspace"]) / "datasets"
        self.cache_key = key
        self.cache = workspace_datasets / key
        self.method_cache_keys = method_keys
        self.method_caches = {m: workspace_datasets / m / m_key for m, m_key in method_keys.items()}
        return {"inputs": len(all_inputs), "cache_key": key, "method_cache_keys": method_keys}

    def datasets(self) -> dict:
        root = self.project / "outputs/Datasets"
        root.mkdir(parents=True, exist_ok=True)
        generation_report = self.project / "outputs/validation/generation_result.json"
        dirs: dict[str, str] = {}
        reused_methods: dict[str, str] = {}

        if self.request["use_cache"]:
            for method in self.required_methods:
                m_cache = self.method_caches[method]
                prov_path = m_cache / "provenance.json"
                if not prov_path.is_file():
                    continue
                provenance = json.loads(prov_path.read_text())
                if (
                    provenance.get("pack") != PACK
                    or provenance.get("method") != method
                    or provenance.get("cache_key") != self.method_cache_keys[method]
                ):
                    raise ValueError(f"Dataset cache identity mismatch for {method}")
                dataset_name = provenance.get("dataset_name")
                if (
                    not isinstance(dataset_name, str)
                    or not dataset_name
                    or "/" in dataset_name
                    or dataset_name in {".", ".."}
                ):
                    raise ValueError(f"Invalid cached dataset name for {method}")
                cached_dir = m_cache / "Datasets" / dataset_name
                if cached_dir.resolve() != cached_dir or not cached_dir.is_dir():
                    raise ValueError(f"Invalid cached dataset directory for {method}")
                current = {
                    p.relative_to(cached_dir).as_posix(): file_hash(p)
                    for p in cached_dir.rglob("*")
                    if p.is_file()
                }
                if not current or current != provenance.get("files"):
                    raise ValueError(f"Dataset cache content hash mismatch for {method}; use --no-cache to regenerate")
                target_dir = root / dataset_name
                _link_cached_files(cached_dir, target_dir)
                dirs[method] = str(target_dir)
                reused_methods[method] = provenance["producer_run_id"]

        missing = [m for m in self.required_methods if m not in dirs]
        if not missing:
            detail = "Reusing hash-verified datasets: " + ", ".join(
                f"{m} from {reused_methods[m]}" for m in self.required_methods
            )
            self.report["checks"].append({"name": "data_generation", "status": "SKIP", "detail": detail})
        else:
            self.phase(
                "data_generation",
                lambda: self.json_command(
                    "data_generation",
                    [
                        self.python,
                        "-B",
                        str(self.project / "scripts/data_processing/generate.py"),
                        "--config",
                        str(self.run / "generation_config.json"),
                        "--method",
                        *missing,
                    ],
                    generation_report,
                    self.config["validation"]["generation_timeout_s"],
                ),
            )
            generated = json.loads(generation_report.read_text())
            gen_dirs = generated.get("datasets", {})
            if set(gen_dirs) != set(missing):
                raise ValueError(f"Generation expected {missing}, produced {sorted(gen_dirs)}")
            for method, value in gen_dirs.items():
                if not Path(value).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Generated dataset path is outside dataset root")
                dirs[method] = value

        if set(dirs) != set(self.required_methods):
            raise ValueError(f"Dataset directories {sorted(dirs)} do not match required {self.required_methods}")
        for value in dirs.values():
            if not Path(value).resolve().is_relative_to(root.resolve()):
                raise ValueError("Dataset path is outside dataset root")

        producer = (
            next(iter(reused_methods.values()))
            if not missing and len(set(reused_methods.values())) == 1
            else self.run.name
        )
        self.report["dataset_cache"] = {
            "reused": not missing,
            "reused_methods": reused_methods,
            "generated_methods": missing,
            "producer_run_id": producer,
            "cache_key": self.cache_key,
            "method_cache_keys": self.method_cache_keys,
        }
        self.dataset_dirs = {m: dirs[m] for m in self.required_methods}
        report_path = self.run / "reports/datasets.json"
        result = self.phase(
            "dataset_contract",
            lambda: self.json_command(
                "dataset_contract",
                [
                    self.python,
                    "-B",
                    str(self.project / "scripts/data_processing/validate.py"),
                    "datasets",
                    "--root",
                    str(root),
                    "--report",
                    str(report_path),
                    "--expected-config",
                    str(self.run / "generation_config.json"),
                    "--methods",
                    *self.required_methods,
                ],
                report_path,
                self.config["validation"]["quick_timeout_s"],
            ),
        )
        if self.request["use_cache"]:
            for method in missing:
                m_cache = self.method_caches[method]
                src_dir = Path(dirs[method])
                dataset_name = src_dir.name
                if m_cache.exists():
                    continue
                staging = m_cache.with_name(f"{m_cache.name}.tmp.{os.getpid()}")
                if staging.exists():
                    shutil.rmtree(staging)
                (staging / "Datasets").mkdir(parents=True)
                shutil.move(str(src_dir), str(staging / "Datasets" / dataset_name))
                cached_target = staging / "Datasets" / dataset_name
                files_hashes = {
                    p.relative_to(cached_target).as_posix(): file_hash(p)
                    for p in cached_target.rglob("*")
                    if p.is_file()
                }
                provenance = {
                    "pack": PACK,
                    "method": method,
                    "cache_key": self.method_cache_keys[method],
                    "producer_run_id": self.run.name,
                    "source_sha256": self.manifest["source_sha256"],
                    "dataset_name": dataset_name,
                    "files": files_hashes,
                }
                write_json(staging / "provenance.json", provenance)
                m_cache.parent.mkdir(parents=True, exist_ok=True)
                staging.replace(m_cache)
                _link_cached_files(m_cache / "Datasets" / dataset_name, src_dir)

        self.report["dataset_directories"] = self.dataset_dirs
        self.save()
        return result

    def train_predict(self, method: str, model: str) -> dict:
        cfg, v = self.config["training"], self.config["validation"]
        data = self.dataset_dirs[method]
        output = self.project / "outputs/networks_results" / method
        entry = str(self.project / entry_script(resolve_selection(method, "complex_CNN")))
        name = f"{method}_{model}"
        train_args = [
            self.python,
            "-B",
            entry,
            "train",
            "--method",
            method,
            "--model",
            model,
            "--data",
            data,
            "--output-dir",
            str(output),
            "--no-resume",
            "--epochs",
            str(cfg["epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            str(cfg["lr"]),
            "--weight-decay",
            str(cfg["weight_decay"]),
            "--base-channels",
            str(cfg["base_channels"]),
            "--dropout",
            str(cfg["dropout"]),
            "--loss-space",
            cfg["loss_space"],
            "--huber-beta",
            str(cfg["huber_beta"]),
            "--val-split-mode",
            cfg["val_split_mode"],
            "--val-fraction",
            str(cfg["val_fraction"]),
            "--val-gap-s",
            str(cfg["val_gap_s"]),
            "--seed",
            str(v["seed"]),
            "--num-workers",
            str(cfg["num_workers"]),
            "--device",
            v["device"],
        ]
        if not cfg["normalize_input"]:
            train_args.append("--no-input-norm")
        self.phase(f"train_{name}", lambda: self.command(f"train_{name}", train_args, v["training_timeout_s"]))
        checkpoints = sorted((output / model / "train_outputs").glob("*_best_*.pt"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected one best checkpoint for {name}, got {checkpoints}")
        histories = list((output / model / "train_outputs").glob("*_history_*.json"))
        if len(histories) != 1:
            raise ValueError(f"Expected one training history for {name}")
        history = json.loads(histories[0].read_text())
        if len(history) != cfg["epochs"] or [row["epoch"] for row in history] != list(range(1, cfg["epochs"] + 1)):
            raise ValueError(f"Unexpected training epoch history for {name}")
        for row in history:
            for key in ("train_loss", "val_loss", "train_mae_km", "val_mae_km", "val_rmse_km"):
                if not math.isfinite(row[key]):
                    raise ValueError(f"Nonfinite training metric {key} for {name}")
        predict_args = [
            self.python,
            "-B",
            entry,
            "predict",
            "--method",
            method,
            "--model",
            model,
            "--data",
            data,
            "--output-dir",
            str(output),
            "--checkpoint",
            str(checkpoints[0]),
            "--batch-size",
            str(cfg["predict_batch_size"]),
            "--num-workers",
            str(cfg["num_workers"]),
            "--device",
            v["device"],
            "--print-first",
            "0",
            "--no-plot",
        ]
        self.phase(f"predict_{name}", lambda: self.command(f"predict_{name}", predict_args, v["training_timeout_s"]))
        csvs = list((output / model / "test_outputs").glob("*_predictions_*.csv"))
        if len(csvs) != 1:
            raise ValueError(f"Expected one prediction CSV for {name}")
        metrics = prediction_metrics(csvs[0], self.test_rows(data))
        return {"method": method, "model": model, "epochs": cfg["epochs"], "checkpoint": str(checkpoints[0]), **metrics}

    def grnn(self, method: str) -> dict:
        g, v = self.config["grnn"], self.config["validation"]
        selection = resolve_selection(method, "GRNN")
        model = selection["model"]
        data = self.dataset_dirs[method]
        output = self.project / "outputs/networks_results" / method
        entry = str(self.project / entry_script(selection))
        build_name, predict_name = f"grnn_{method}_build", f"grnn_{method}_predict"
        build_args = [
            self.python,
            "-B",
            entry,
            "build",
            "--method",
            method,
            "--data",
            data,
            "--output-dir",
            str(output),
            "--cv-folds",
            str(g["cv_folds"]),
            "--spread",
            str(g["spread"]),
        ]
        if g["sigma_candidates"]:
            build_args.extend(["--sigma-candidates", *map(str, g["sigma_candidates"])])
        build_args.extend([
            "--seed",
            str(v["seed"]),
            "--batch-size",
            str(g["batch_size"]),
            "--device",
            v["device"],
        ])
        if g["standardize_input"]:
            build_args.append("--standardize-input")
        if g["per_sample_channel_norm"]:
            build_args.append("--per-sample-channel-norm")
        if g["max_reference_samples"] > 0:
            build_args.extend(["--max-reference-samples", str(g["max_reference_samples"])])
        if g["max_validation_samples"] > 0:
            build_args.extend(["--max-validation-samples", str(g["max_validation_samples"])])
        self.phase(build_name, lambda: self.command(build_name, build_args, v["training_timeout_s"]))
        references = list((output / model / "reference_outputs").glob("*_reference_*.pt"))
        if len(references) != 1:
            raise ValueError("Expected one GRNN reference artifact")
        predict_args = [
            self.python,
            "-B",
            entry,
            "predict",
            "--method",
            method,
            "--data",
            data,
            "--output-dir",
            str(output),
            "--reference",
            str(references[0]),
            "--batch-size",
            str(g["predict_batch_size"]),
            "--device",
            v["device"],
            "--print-first",
            "0",
            "--no-plot",
        ]
        if g["max_predict_samples"] > 0:
            predict_args.extend(["--max-predict-samples", str(g["max_predict_samples"])])
        self.phase(predict_name, lambda: self.command(predict_name, predict_args, v["training_timeout_s"]))
        csvs = list((output / model / "test_outputs").glob("*_predictions_*.csv"))
        if len(csvs) != 1:
            raise ValueError("Expected one GRNN prediction CSV")
        expected_test = self.test_rows(data)
        if g["max_predict_samples"] > 0:
            expected_test = expected_test[: g["max_predict_samples"]]
        return {"method": method, "model": model, "reference": str(references[0]), **prediction_metrics(csvs[0], expected_test)}

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
        plan = model_plan(self.request["suite"], self.request.get("selection"))
        job = next((item for item in plan if item["architecture"] != "GRNN"), None)
        if job is None:
            raise ValueError("Checkpoint resume requires at least one neural model in the execution plan")
        method, model = job["method"], job["model"]
        cfg, v = self.config["training"], self.config["validation"]
        source = self.project / "outputs/networks_results" / method / model / "train_outputs"
        last = list(source.glob("*_last_*.pt"))
        if len(last) != 1:
            raise ValueError("Expected one last checkpoint for resume validation")
        target = self.project / "outputs/resume_validation" / method
        resume_args = [
            self.python,
            "-B",
            str(self.project / entry_script(job)),
            "train",
            "--method",
            method,
            "--model",
            model,
            "--data",
            self.dataset_dirs[method],
            "--output-dir",
            str(target),
            "--resume-checkpoint",
            str(last[0]),
            "--epochs",
            "1",
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            str(cfg["lr"]),
            "--weight-decay",
            str(cfg["weight_decay"]),
            "--base-channels",
            str(cfg["base_channels"]),
            "--dropout",
            str(cfg["dropout"]),
            "--loss-space",
            cfg["loss_space"],
            "--huber-beta",
            str(cfg["huber_beta"]),
            "--val-split-mode",
            cfg["val_split_mode"],
            "--val-fraction",
            str(cfg["val_fraction"]),
            "--val-gap-s",
            str(cfg["val_gap_s"]),
            "--seed",
            str(v["seed"]),
            "--num-workers",
            str(cfg["num_workers"]),
            "--device",
            v["device"],
        ]
        if not cfg["normalize_input"]:
            resume_args.append("--no-input-norm")
        self.command("checkpoint_resume", resume_args, v["training_timeout_s"])
        import torch
        resumed_paths = list(target.rglob("*_last_*.pt"))
        if len(resumed_paths) != 1:
            raise ValueError("Resume validation did not produce one checkpoint")
        resumed = torch.load(resumed_paths[0], map_location="cpu", weights_only=True)
        original = torch.load(last[0], map_location="cpu", weights_only=True)
        if resumed["epoch"] != original["epoch"] + 1 or resumed["train_indices"] != original["train_indices"] or resumed["val_indices"] != original["val_indices"]:
            raise ValueError("Resumed epoch/split differs from expected checkpoint")

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
        self.command("visualization", [self.python, "-B", str(self.project / "scripts/remote_workflow/render.py"),
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
            if suite != "selected":
                python_report = self.run / "reports/python_quick.json"
                self.phase("python_quick", lambda: self.json_command("python_quick", [self.python, "-B", str(self.project / "scripts/remote_workflow/validation/validate.py"), "quick", "--device", v["device"], "--report", str(python_report)], python_report, v["quick_timeout_s"]))
                numerical_report = self.project / "outputs/validation/data_numerical.json"
                self.phase("data_numerical", lambda: self.json_command("data_numerical", [self.python, "-B", str(self.project / "scripts/data_processing/validate.py"), "--report", str(numerical_report)], numerical_report, v["quick_timeout_s"]))
            if suite != "quick":
                self.phase("original_data_identity", self.data_identity)
                self.datasets()
                metrics = []
                plan = model_plan(suite, selection)
                for job in plan:
                    method, model = job["method"], job["model"]
                    if job["architecture"] == "GRNN":
                        metrics.append(self.phase(f"evaluate_{method}_{model}", lambda m=method: self.grnn(m)))
                    else:
                        metrics.append(self.phase(f"evaluate_{method}_{model}", lambda m=method, n=model: self.train_predict(m, n)))
                indices_by_method: dict[str, list[int]] = {}
                for row in metrics:
                    expected_indices = indices_by_method.setdefault(row["method"], row["source_segment_indices"])
                    if row["source_segment_indices"] != expected_indices:
                        raise ValueError(f"{row['method']} models did not evaluate the same test source windows")
                if suite in {"smoke", "experiment"} and any(item["architecture"] != "GRNN" for item in plan):
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
            self.report["not_run"] = sorted(planned_phases - completed)
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
