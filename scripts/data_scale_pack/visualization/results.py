"""Read one recorded run without importing numerical or training libraries."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.run_identity import validate_run_identity

PREDICTION_FIELDS = (
    "sample", "source_segment_idx", "window_center_s", "pred_range_km",
    "true_range_km", "abs_error_km",
)
TIME_TOLERANCE = 1.1e-6
ERROR_TOLERANCE = 2.1e-6
PACK = "data_scale_pack"


def recorded_contract(report: dict, run_dir: Path, sources: list[dict]) -> tuple[list[dict], dict]:
    """Bind every rendered result to its complete frozen source/configuration plan."""
    planned = model_plan(report["suite"], report.get("selection"))
    if report.get("model_plan") != planned:
        raise ValueError("Recorded model plan differs from its complete canonical suite plan")
    manifest = _json(_read(run_dir / "manifest.json", run_dir, sources))
    request = _json(_read(run_dir / "request.json", run_dir, sources))
    for record in (manifest, request):
        if not isinstance(record, dict) or any(record.get(key) != report.get(key) for key in ("pack", "run_id", "suite")):
            raise ValueError("Manifest/request pack, run or suite differs from report")
        if record.get("selection") != report.get("selection"):
            raise ValueError("Manifest/request selection differs from report")
    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory or any(not isinstance(item, dict) for item in inventory):
        raise ValueError("Run requires a recorded source inventory")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("source_sha256") or digest != report.get("source_sha256"):
        raise ValueError("Source inventory hash differs from manifest/report")
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("Run requires its frozen configuration")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("config_sha256") or digest != report.get("config_sha256"):
        raise ValueError("Configuration hash differs from manifest/report")
    validate_run_identity(report["run_id"], PACK, config, report.get("selection"))
    return planned, config


@dataclass(frozen=True)
class Prediction:
    sample: int
    source_index: int
    time_s: float
    predicted_km: float
    true_km: float

    @property
    def residual_km(self) -> float:
        return self.predicted_km - self.true_km


@dataclass
class ModelResult:
    method: str
    model: str
    predictions: list[Prediction]
    history: list[dict]
    metrics: dict
    architecture: dict

    @property
    def name(self) -> str:
        return f"{self.method}/{self.model}"

    @property
    def label(self) -> str:
        scaling = self.architecture["scaling"]
        return f"{self.method}/{scaling['selection_mode']} {scaling['selected_profile']}"

    @property
    def caption(self) -> str:
        scaling = self.architecture["scaling"]
        return f"{self.label}\nNtrain={scaling['train_count']:,}; P={scaling['parameter_count']:,}"



@dataclass
class RunResults:
    run_dir: Path
    report: dict
    report_bytes: bytes
    models: list[ModelResult]
    sources: list[dict]
    config: dict


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
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != PREDICTION_FIELDS:
        raise ValueError("Prediction CSV must have the canonical six columns")
    predictions = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed prediction CSV row")
        try:
            sample, index = int(row["sample"]), int(row["source_segment_idx"])
            moment, pred, target, error = (
                float(row[key]) for key in PREDICTION_FIELDS[2:]
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid prediction value") from exc
        if not all(math.isfinite(x) for x in (moment, pred, target, error)):
            raise ValueError("Nonfinite prediction value")
        if sample != len(predictions) or index < 1:
            raise ValueError("Invalid sample/source window index")
        if predictions and (index <= predictions[-1].source_index or moment <= predictions[-1].time_s):
            raise ValueError("Source windows and timestamps must strictly increase")
        if error < 0 or abs(error - abs(pred-target)) > ERROR_TOLERANCE:
            raise ValueError("Reported absolute error differs from prediction and label")
        predictions.append(Prediction(sample, index, moment, pred, target))
    if not predictions:
        raise ValueError("Empty prediction CSV")
    return predictions


def summarize(rows: list[Prediction]) -> dict:
    residuals = [row.residual_km for row in rows]
    errors = [abs(value) for value in residuals]
    worst = max(range(len(rows)), key=errors.__getitem__)
    return {"samples": len(rows), "mae_km": math.fsum(errors)/len(rows),
            "rmse_km": math.sqrt(math.fsum(value*value for value in errors)/len(rows)),
            "bias_km": math.fsum(residuals)/len(rows), "max_error_km": errors[worst],
            "max_error_window_center_s": rows[worst].time_s,
            "source_segment_indices": [row.source_index for row in rows],
            "first_window_center_s": rows[0].time_s, "last_window_center_s": rows[-1].time_s}


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


def read_architecture(raw: bytes, row: dict, job: dict, config: dict) -> dict:
    architecture = _json(raw)
    if (not isinstance(architecture, dict) or architecture.get("pack") != PACK
            or architecture.get("method") != job["method"] or architecture.get("model") != job["model"]):
        raise ValueError("Architecture sidecar belongs to another pack or model")
    scaling = architecture.get("scaling")
    if not isinstance(scaling, dict) or scaling != row.get("scaling"):
        raise ValueError("Architecture scaling metadata differs from the report")
    required = {"train_count", "scaling_policy_version", "scaling_policy", "selection_mode", "selected_profile",
                "stage_repeats", "stage_strides", "parameter_count", "parameter_budget", "calibration_status"}
    if set(scaling) != required:
        raise ValueError("Architecture scaling metadata is incomplete or contains unknown fields")
    for key in ("train_count", "parameter_count"):
        if type(scaling[key]) is not int or scaling[key] < 1:
            raise ValueError(f"Architecture {key} must be a positive integer")
    if finite(scaling["parameter_budget"], "architecture.parameter_budget") <= 0:
        raise ValueError("Architecture parameter budget must be positive")
    mode = "auto" if job["architecture"] == "complex_CNN" else "fixed"
    profile = scaling["selected_profile"]
    if scaling["selection_mode"] != mode or profile not in {"small", "medium", "large", "xlarge", "baseline"}:
        raise ValueError("Architecture profile/selection mode is invalid")
    if (mode == "auto" and profile == "baseline") or (mode == "fixed" and profile != job["architecture"].removeprefix("complex_CNN_")):
        raise ValueError("Architecture profile differs from its planned model")
    if scaling["scaling_policy_version"] != "train-count-v1" or scaling["calibration_status"] != "uncalibrated":
        raise ValueError("Unknown scaling policy or calibration status")
    policy = scaling["scaling_policy"]
    if not isinstance(policy, dict) or set(policy) != {"reference_samples", "exponent"}:
        raise ValueError("Invalid scaling policy metadata")
    if policy != config["scaling"] or type(policy["reference_samples"]) is not int:
        raise ValueError("Architecture scaling policy differs from frozen configuration")
    for key, value in policy.items():
        if finite(value, f"scaling_policy.{key}") <= 0:
            raise ValueError("Scaling policy values must be positive")
    if not isinstance(scaling["stage_repeats"], list) or not scaling["stage_repeats"] or any(type(x) is not int or x < 1 for x in scaling["stage_repeats"]):
        raise ValueError("Invalid stage repetition metadata")
    if not isinstance(scaling["stage_strides"], list) or len(scaling["stage_strides"]) != len(scaling["stage_repeats"]):
        raise ValueError("Invalid stage stride metadata")
    if any(not isinstance(stride, list) or len(stride) != 2 or any(type(x) is not int or x < 1 for x in stride) for stride in scaling["stage_strides"]):
        raise ValueError("Invalid stage stride dimensions")
    if not isinstance(architecture.get("model_config"), dict) or not architecture["model_config"]:
        raise ValueError("Architecture sidecar is missing its model configuration")
    model_config = architecture["model_config"]
    shape_field = "input_elements" if job["method"] == "RBD" else "input_pairs"
    expected_config = required - {"calibration_status"} | {"base_channels", "dropout", shape_field, "input_freq_bins"}
    if set(model_config) != expected_config or any(model_config[key] != scaling[key] for key in required - {"calibration_status"}):
        raise ValueError("Model configuration disagrees with frozen scaling metadata")
    for key in ("base_channels", shape_field, "input_freq_bins"):
        if type(model_config[key]) is not int or model_config[key] < 1:
            raise ValueError(f"Model configuration {key} must be a positive integer")
    if (model_config["base_channels"] != config["training"]["base_channels"]
            or not 0 <= finite(model_config["dropout"], "model_config.dropout") < 1):
        raise ValueError("Model configuration width or dropout is invalid")
    sampling = architecture.get("sampling")
    fraction = finite(row.get("train_fraction"), "report.train_fraction")
    if not 0 < fraction <= 1 or not isinstance(sampling, dict) or sampling.get("train_fraction") != fraction:
        raise ValueError("Architecture sampling differs from the reported training fraction")
    if fraction != config["training"]["train_fraction"]:
        raise ValueError("Architecture training fraction differs from frozen configuration")
    required_sampling = {"version", "train_fraction", "candidate_train_count", "train_count", "validation_count",
                         "block_rows", "seed", "source_identity_sha256"}
    if set(sampling) != required_sampling or sampling["version"] != "temporal-block-prefix-v1":
        raise ValueError("Unknown or incomplete training sampling metadata")
    for key in ("candidate_train_count", "train_count", "validation_count", "block_rows"):
        if type(sampling[key]) is not int or sampling[key] < 1:
            raise ValueError(f"Sampling {key} must be a positive integer")
    if (sampling["train_count"] != scaling["train_count"] or sampling["candidate_train_count"] < sampling["train_count"]
            or sampling["block_rows"] != 32 or type(sampling["seed"]) is not int
            or sampling["seed"] != config["validation"]["seed"]
            or not isinstance(sampling["source_identity_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", sampling["source_identity_sha256"])):
        raise ValueError("Sampling identities/counts differ from the training contract")
    return architecture


def load_run(run_dir: Path, models: list[str] | None = None, *, require_pass=True) -> RunResults:
    run_dir = run_dir.resolve()
    sources = []
    raw_report = _read(run_dir/"report.json", run_dir, sources)
    report = _json(raw_report)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Unsupported run report")
    if report.get("pack") != PACK:
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
        if not isinstance(row, dict) or row.get("method") not in {"RBD", "ELM", "SCM"} or not re.fullmatch(r"[a-z][a-z0-9_]*", str(row.get("model", ""))):
            raise ValueError("Invalid method/model in report")
        names.append(f"{row['method']}/{row['model']}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate method/model in run report")
    planned, config = recorded_contract(report, run_dir, sources)
    jobs = {(job["method"], job["model"]): job for job in planned}
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
        phase = f"evaluate_{method}_{model}"
        if phase not in completed:
            raise ValueError(f"Model evaluation has not passed: {phase}")
        directory = root/method/model
        prediction_path = _unique(directory/"test_outputs", "*_predictions_*.csv")
        predictions = read_predictions(_read(prediction_path, run_dir, sources))
        summary = summarize(predictions)
        _verify_metrics(summary, row)
        epochs = row.get("epochs")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("Reported training epoch count must be a positive integer")
        path = _unique(directory/"train_outputs", "*_history_*.json")
        history = read_history(_read(path, run_dir, sources), epochs)
        path = _unique(directory/"train_outputs", "*_architecture_*.json")
        if row.get("architecture_path") != path.relative_to(run_dir).as_posix():
            raise ValueError("Reported architecture path differs from its canonical model output")
        architecture = read_architecture(_read(path, run_dir, sources), row, jobs[(method, model)], config)
        results.append(ModelResult(method, model, predictions, history, summary, architecture))

    base = results[0].predictions
    for result in results[1:]:
        rows = result.predictions
        if len(rows) != len(base) or any(
            a.source_index != b.source_index or abs(a.time_s-b.time_s) > TIME_TOLERANCE
            or abs(a.true_km-b.true_km) > TIME_TOLERANCE for a, b in zip(base, rows)
        ):
            raise ValueError("Models must use identical test source windows, times and labels")
    if models is not None:
        # Filtering only controls rendering, after the whole recorded suite has
        # been validated. Missing or damaged unselected inputs must still fail.
        results = [result for result in results if result.name in models]
    return RunResults(run_dir, report, raw_report, results, sources, config)
