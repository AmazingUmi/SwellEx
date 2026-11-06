"""Read one recorded run without importing numerical or training libraries."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.network_training.grnn.bindings import METHODS, model_name
from scripts.remote_workflow.metrics_contract import (
    ERROR_TOLERANCE,
    PREDICTION_FIELDS,
    PredictionRow,
    TIME_TOLERANCE,
    compute_prediction_metrics,
    parse_prediction_bytes,
)
from scripts.remote_workflow.model_plan import model_plan, validate_selection
from scripts.remote_workflow.packs import DEFAULT_PACK

Prediction = PredictionRow
# This recorded source version predates RBD GRNN. Historical reports do
# not carry model_plan, so their manifest must identify that exact contract.
LEGACY_TASKS_SHA256 = "3457c6a2ae05096af371a74d2744cb6d4cbbf8132ec6365d7480035c38dd93eb"
# Historical manifests retain their pre-pack source paths; this is evidence
# lookup only and is never used as an executable entry or import alias.
TASKS_SOURCE = "scripts/localization_networks/workflow/tasks.py"


def is_grnn(method: str, model: str) -> bool:
    return method in METHODS and model == model_name(method)


def recorded_model_plan(report: dict, run_dir: Path, sources: list[dict]) -> tuple[list[dict], bool]:
    """Validate the source-bound execution contract, including saved old runs.

    Newly recorded results require the explicit current plan. Only a manifest
    identifying the known SCM-only source contract enables historical reading.
    This is result analysis, never an execution alias or a partial-suite fallback.
    """
    planned = model_plan(report["suite"], report.get("selection"))
    if "model_plan" in report:
        if report["model_plan"] != planned:
            raise ValueError("Recorded model plan differs from its complete canonical suite plan")
        return planned, False
    manifest = _json(_read(run_dir/"manifest.json", run_dir, sources))
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or any(not isinstance(item, dict) for item in inventory):
        raise ValueError("Historical run requires a recorded source inventory")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tasks = [item for item in inventory if item.get("path") == TASKS_SOURCE]
    if (manifest.get("run_id") != report["run_id"] or manifest.get("suite") != report["suite"]
            or digest != manifest.get("source_sha256") or digest != report.get("source_sha256")
            or len(tasks) != 1 or tasks[0].get("sha256") != LEGACY_TASKS_SHA256):
        raise ValueError("Missing model plan without the known historical source contract")
    legacy = [item for item in planned if item["architecture"] != "GRNN" or item["method"] == "SCM"]
    if not legacy:
        raise ValueError("The historical source contract supports GRNN only for SCM")
    if report["suite"] == "selected" and manifest.get("selection") != report.get("selection"):
        raise ValueError("Historical manifest selection differs from the selected run")
    return legacy, True


@dataclass
class ModelResult:
    method: str
    model: str
    predictions: list[Prediction]
    history: list[dict]
    sigma_search: list[dict]
    metrics: dict

    @property
    def name(self) -> str:
        return f"{self.method}/{self.model}"


@dataclass
class RunResults:
    run_dir: Path
    report: dict
    report_bytes: bytes
    models: list[ModelResult]
    sources: list[dict]


def finite(value, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path: Path, root: Path, sources: list[dict]) -> bytes:
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or external result file: {path}")
    raw = path.read_bytes()
    sources.append({"path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
    return raw


def _json(raw: bytes):
    return json.loads(raw, object_pairs_hook=_object)


def _unique(directory: Path, pattern: str, *, optional=False) -> Path | None:
    paths = sorted(directory.glob(pattern))
    if optional and not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}; found {len(paths)}. Select a single run.")
    return paths[0]


def read_predictions(raw: bytes) -> list[Prediction]:
    return parse_prediction_bytes(raw)


def summarize(rows: list[Prediction]) -> dict:
    return compute_prediction_metrics(rows)


def _verify_metrics(actual: dict, expected: dict) -> None:
    if type(expected.get("samples")) is not int or expected["samples"] != actual["samples"]:
        raise ValueError("Prediction count differs from run report")
    indices = expected.get("source_segment_indices")
    if not isinstance(indices, list) or any(type(x) is not int for x in indices) or indices != actual["source_segment_indices"]:
        raise ValueError("Prediction source windows differ from run report")
    for key in ("mae_km", "rmse_km", "max_error_km", "max_error_window_center_s"):
        value = finite(expected.get(key), f"report.{key}", nonnegative=key != "max_error_window_center_s")
        if not math.isclose(value, actual[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Recomputed {key} differs from run report")


def read_history(raw: bytes, epochs: int) -> list[dict]:
    history = _json(raw)
    if not isinstance(history, list) or not history or len(history) != epochs:
        raise ValueError("Training history count differs from reported epochs")
    loss_space = None
    for index, row in enumerate(history, 1):
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] != index:
            raise ValueError("Training epochs must be continuous, starting at 1")
        mode = row.get("loss_space")
        if mode not in {"km", "normalized"} or (loss_space is not None and mode != loss_space):
            raise ValueError("Training loss space is missing or changes between epochs")
        loss_space = mode
        for key in ("lr", "train_loss", "val_loss", "train_mae_km", "val_mae_km", "train_rmse_km", "val_rmse_km"):
            finite(row.get(key), f"history.{key}", nonnegative=True)
    return history


def read_sigma_search(raw: bytes) -> list[dict]:
    records = _json(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Empty GRNN sigma search")
    mode = None
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Invalid GRNN sigma record")
        prefix = "mean_" if "mean_mape_pct" in row else ""
        if mode is not None and prefix != mode:
            raise ValueError("GRNN search mixes CV and validation metrics")
        mode = prefix
        sigma = finite(row.get("sigma"), "sigma")
        if sigma <= 0 or sigma in seen:
            raise ValueError("GRNN sigma must be unique and positive")
        seen.add(sigma)
        for key in ("mae_km", "rmse_km", "mape_pct"):
            finite(row.get(prefix+key), "sigma_search."+prefix+key, nonnegative=True)
    return records


def load_run(run_dir: Path, models: list[str] | None = None, *, require_pass=True) -> RunResults:
    run_dir = run_dir.resolve()
    sources = []
    raw_report = _read(run_dir/"report.json", run_dir, sources)
    report = _json(raw_report)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Unsupported run report")
    if report.get("pack", DEFAULT_PACK) != DEFAULT_PACK:
        raise ValueError("Visualization report belongs to another experiment pack")
    allowed = {"PASS"} if require_pass else {"PASS", "RUNNING"}
    if report.get("status") not in allowed or report.get("suite") not in {"selected", "smoke", "experiment"}:
        raise ValueError("Visualization requires a successful selected/smoke/experiment run with model results")
    if report.get("run_id") != run_dir.name or not isinstance(report.get("purpose"), str):
        raise ValueError("Run identity/purpose differs from result directory")
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Run has no model results")
    if report["suite"] == "selected":
        if report["purpose"] != "configured_model_selection_without_accuracy_acceptance_threshold":
            raise ValueError("Selected run purpose is inconsistent")
        planned_selection = model_plan("selected", report.get("selection"))
        metrics = report.get("metrics")
        expected = {(job["method"], job["model"]) for job in planned_selection}
        if (not isinstance(metrics, list) or len(metrics) != len(expected)
                or any(not isinstance(row, dict) for row in metrics)
                or {(row.get("method"), row.get("model")) for row in metrics} != expected):
            raise ValueError("Selected run metrics differ from its complete filtered plan")
    names = []
    for row in metrics:
        if not isinstance(row, dict) or row.get("method") not in {"RBD", "SCM"} or not re.fullmatch(r"[a-z][a-z0-9_]*", str(row.get("model", ""))):
            raise ValueError("Invalid method/model in report")
        names.append(f"{row['method']}/{row['model']}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate method/model in run report")
    planned, historical = recorded_model_plan(report, run_dir, sources)
    expected_names = {f"{item['method']}/{item['model']}" for item in planned}
    if set(names) != expected_names:
        raise ValueError("Run model results differ from its complete canonical suite plan")
    if models is not None and (not models or len(models) != len(set(models)) or set(models)-set(names)):
        raise ValueError(f"Unknown or duplicate model selection; available: {', '.join(names)}")
    checks = report.get("checks", [])
    if not isinstance(checks, list) or any(not isinstance(c, dict) or c.get("status") == "FAIL" for c in checks):
        raise ValueError("Run contains a failed validation phase")
    completed = {c.get("name") for c in checks if c.get("status") == "PASS"}
    results = []
    root = run_dir/"project/outputs/networks_results"
    for row, name in zip(metrics, names):
        method, model = row["method"], row["model"]
        grnn = is_grnn(method, model)
        phase = "evaluate_grnn" if historical and grnn else f"evaluate_{method}_{model}"
        if phase not in completed:
            raise ValueError(f"Model evaluation has not passed: {phase}")
        directory = root/method/model
        prediction_path = _unique(directory/"test_outputs", "*_predictions_*.csv")
        predictions = read_predictions(_read(prediction_path, run_dir, sources))
        summary = summarize(predictions)
        _verify_metrics(summary, row)
        history, sigma_search = [], []
        if grnn:
            search = _unique(directory/"reference_outputs", "*_sigma_search_*.json", optional=True)
            if search:
                sigma_search = read_sigma_search(_read(search, run_dir, sources))
        else:
            epochs = row.get("epochs")
            if type(epochs) is not int or epochs < 1:
                raise ValueError("Reported training epoch count must be a positive integer")
            path = _unique(directory/"train_outputs", "*_history_*.json")
            history = read_history(_read(path, run_dir, sources), epochs)
        results.append(ModelResult(method, model, predictions, history, sigma_search, summary))
    base_by_method: dict[str, list[Prediction]] = {}
    for result in results:
        base = base_by_method.setdefault(result.method, result.predictions)
        rows = result.predictions
        if len(rows) != len(base) or any(
            a.source_index != b.source_index or abs(a.time_s-b.time_s) > TIME_TOLERANCE
            or abs(a.true_km-b.true_km) > TIME_TOLERANCE for a, b in zip(base, rows)
        ):
            raise ValueError(f"Models for {result.method} must use identical test source windows, times and labels")
    if models is not None:
        # Filtering only controls rendering, after the whole recorded suite has
        # been validated. Missing or damaged unselected inputs must still fail.
        results = [result for result in results if result.name in models]
    return RunResults(run_dir, report, raw_report, results, sources)
