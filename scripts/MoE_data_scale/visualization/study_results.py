"""Strict, standard-library evidence summaries for fixed-expert MoE matrices.

Only complete seed groups receive means/sample SDs. Test values never select E.
No model execution, numerical dependencies, or matrix plots are imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scripts.MoE_data_scale.visualization.results import _json, finite, load_run
from scripts.run_identity import validate_run_identity

PACK = "MoE_data_scale"
TRAINING_METRICS = (
    "best_epoch", "best_val_rmse_km", "best_val_mae_km", "best_train_loss",
    "best_train_mae_km", "best_train_rmse_km", "total_train_wall_s",
    "sum_epoch_elapsed_s", "optimizer_steps",
)
ESTIMANDS = (
    "test_mae_km", "test_rmse_km", "best_val_rmse_km", "best_val_mae_km",
    "best_train_mae_km", "best_train_rmse_km", "best_epoch", "train_count",
    "parameter_count", "active_parameter_count", "total_train_wall_s",
    "sum_epoch_elapsed_s", "optimizer_steps", "best_train_router_entropy",
    "best_val_router_entropy",
)
LIMITATIONS = [
    "Fixed-expert comparisons do not calibrate the automatic expert-count rule.",
    "Only complete planned seed groups have arithmetic means and sample SDs (ddof=1); missing results are never zero or imputed.",
    "Seed variation includes split/model randomness, not independent environments; source windows remain temporally correlated.",
    "Expert count is selected using mean best validation RMSE only; test metrics do not participate in selection.",
    "Routing load and entropy come from the validation-selected epoch, with train and validation explicitly distinguished.",
    "Training metrics accumulate online while weights update; they are not frozen-model training scores.",
    "Wall time records the run environment and is not a controlled throughput benchmark.",
    "Fixed epochs permit different update counts across data fractions; no significance inference is performed.",
]


def digest(value, *, compact=False) -> str:
    options = {"sort_keys": True, "allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def split_identity(rows) -> str:
    ordered = sorted(rows, key=lambda row: row[0])
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path: Path, sources: list[dict], root: Path | None = None) -> object:
    if not path.is_file() or path.is_symlink() or (root is not None and not path.resolve().is_relative_to(root.resolve())):
        raise ValueError(f"Missing or symlink evidence: {path}")
    raw = path.read_bytes()
    sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
    return _json(raw)


def verify_split(value: object, name: str) -> dict[int, tuple[float, float]]:
    if not isinstance(value, dict) or type(value.get("count")) is not int or value["count"] < 1:
        raise ValueError(f"{name}: missing nonempty split inventory")
    keys = ("source_segment_indices", "window_center_s", "range_km")
    if any(not isinstance(value.get(key), list) or len(value[key]) != value["count"] for key in keys):
        raise ValueError(f"{name}: split members differ from count")
    members = {}
    for source, moment, distance in zip(*(value[key] for key in keys), strict=True):
        if type(source) is not int or source < 1 or source in members:
            raise ValueError(f"{name}: source identities must be unique positive integers")
        members[source] = (finite(moment, name + ".time"), finite(distance, name + ".range"))
    identity = split_identity([(source, *values) for source, values in members.items()])
    if value.get("identity_sha256") != identity:
        raise ValueError(f"{name}: source identity hash mismatch")
    times = [pair[0] for pair in members.values()]
    distances = [pair[1] for pair in members.values()]
    if value.get("time_s") != {"min": min(times), "max": max(times)}:
        raise ValueError(f"{name}: time coverage differs from members")
    summary = value.get("range_km_summary")
    if not isinstance(summary, dict) or summary.get("min") != min(distances) or summary.get("max") != max(distances):
        raise ValueError(f"{name}: range coverage differs from members")
    ordered = [finite(summary.get(key), name + "." + key) for key in ("min", "q25", "median", "q75", "max")]
    if ordered != sorted(ordered):
        raise ValueError(f"{name}: unordered range summaries")
    return members


def _same_members(left: dict, right: dict, context: str) -> None:
    if left != right:
        raise ValueError(context + ": physical sample identities differ")


def _verify_training(row: dict, result, entry: dict) -> dict:
    for key in TRAINING_METRICS:
        value = row.get(key)
        if key in {"best_epoch", "optimizer_steps"}:
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid reported {key}")
        else:
            finite(value, "reported." + key, nonnegative=True)
    history = result.history
    best = min(history, key=lambda item: item["val_rmse_km"])
    if row["best_epoch"] != best["epoch"]:
        raise ValueError("Reported best epoch is not the first minimum-validation-RMSE checkpoint")
    fields = {"best_val_rmse_km": "val_rmse_km", "best_val_mae_km": "val_mae_km",
              "best_train_loss": "train_loss", "best_train_mae_km": "train_mae_km", "best_train_rmse_km": "train_rmse_km"}
    for target, source in fields.items():
        if not math.isclose(row[target], best[source], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Reported {target} differs from best-checkpoint history")
    elapsed, steps = 0.0, 0
    for epoch in history:
        elapsed += finite(epoch.get("epoch_elapsed_s"), "history.epoch_elapsed_s", nonnegative=True)
        count = epoch.get("optimizer_steps_epoch")
        if type(count) is not int or count < 1:
            raise ValueError("History is missing per-epoch optimizer step counts")
        steps += count
        if epoch.get("optimizer_steps") != steps:
            raise ValueError("History cumulative optimizer steps are inconsistent")
    if elapsed > row["total_train_wall_s"] + 0.01:
        raise ValueError("Epoch duration exceeds enclosing training wall time")
    if steps != row["optimizer_steps"] or not math.isclose(elapsed, row["sum_epoch_elapsed_s"], rel_tol=1e-8, abs_tol=1e-7):
        raise ValueError("Reported training cost differs from complete epoch history")
    recorded = entry.get("training_metrics")
    if (not isinstance(recorded, dict) or not set(TRAINING_METRICS).issubset(recorded)
            or any(value != row.get(key) for key, value in recorded.items())):
        raise ValueError("Training inventory summary differs from reported training metrics")
    return {key: row[key] for key in TRAINING_METRICS}


@dataclass
class StudyResults:
    study_path: Path
    study: dict
    rows: list[dict]
    summaries: list[dict]
    sources: list[dict]

    @property
    def status(self) -> str:
        return "PASS" if self.rows and all(row["status"] == "PASS" for row in self.rows) else "PARTIAL"


def summarize_rows(rows: list[dict], seeds: list[int]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["train_fraction"], row["experts"]), []).append(row)
    summaries = []
    for (method, fraction, experts), group in groups.items():
        valid = [row for row in group if row["status"] == "PASS"]
        observed = [row["seed"] for row in valid]
        complete = sorted(observed) == sorted(seeds) and len(observed) == len(set(observed))
        summary = {"method": method, "train_fraction": fraction, "experts": experts,
                   "status": "PASS" if complete else "PARTIAL", "n_seeds": len(valid),
                   "planned_seeds": len(seeds), "observed_seeds": observed,
                   "missing_seeds": [seed for seed in seeds if seed not in observed]}
        for key in ESTIMANDS:
            values = [finite(row[key], key) for row in valid] if complete else []
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_sd"] = statistics.stdev(values) if len(values) >= 2 else None
        for split in ("train", "val"):
            # Expert ordering is fixed within an E condition; no pooling across E.
            loads = [row[f"best_{split}_expert_load"] for row in valid] if complete else []
            summary[f"best_{split}_expert_load_mean"] = [statistics.mean(items) for items in zip(*loads)] if loads else None
            summary[f"best_{split}_expert_load_sd"] = [statistics.stdev(items) for items in zip(*loads)] if len(loads) >= 2 else None
        summaries.append(summary)
    return summaries


def _source_inventory(value: object) -> dict:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Missing training source byte identity")
    normalized = {}
    for field in ("train_sources", "validation_sources"):
        items = value.get(field)
        if not isinstance(items, list) or not items:
            raise ValueError("Missing training source byte inventory")
        if any(not isinstance(item, dict) or type(item.get("size_bytes")) is not int or item["size_bytes"] < 1
               or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in items):
            raise ValueError("Invalid training source byte size/hash")
        normalized[field] = [{key: item[key] for key in ("size_bytes", "sha256")} for item in items]
    return normalized


def _verify_model_evidence(result, entry: dict, seed: int, fraction: float, experts: int) -> tuple[dict, dict, dict, dict]:
    if len(result.models) != 1 or len(result.report["metrics"]) != 1:
        raise ValueError("Each matrix child must contain exactly one selected model")
    model, row = result.models[0], result.report["metrics"][0]
    if (entry.get("method"), entry.get("model"), entry.get("seed"), entry.get("train_fraction")) != (model.method, model.model, seed, fraction):
        raise ValueError("Training inventory model/seed/fraction differs from study cell")
    if row.get("training_inventory_path") != "reports/training_inventory.json":
        raise ValueError("Training inventory path differs from canonical evidence")
    if len(model.history) != result.config["training"]["epochs"]:
        raise ValueError("Training history differs from frozen epoch budget")
    inventory = entry.get("training_inventory")
    if not isinstance(inventory, dict) or inventory.get("schema_version") != 1 or inventory != model.architecture.get("training_inventory"):
        raise ValueError("Sidecar and training inventory report disagree")
    training = verify_split(inventory.get("train"), "training")
    validation = verify_split(inventory.get("validation"), "validation")
    testing = verify_split(entry.get("test"), "test")
    if set(training) & (set(validation) | set(testing)) or set(validation) & set(testing):
        raise ValueError("Training/validation/test physical identities overlap")
    scaling, sampling = model.architecture["scaling"], model.architecture["sampling"]
    if (scaling["num_experts"] != experts or scaling["selection_mode"] != "fixed"
            or len(training) != scaling["train_count"] or len(validation) != sampling["validation_count"]
            or len(training) != math.floor(sampling["candidate_train_count"] * fraction)
            or entry.get("sampling") != sampling):
        raise ValueError("Actual training budget/expert count differs from frozen study cell")
    if row.get("parameter_count") != scaling["parameter_count"]:
        raise ValueError("Parameter count differs from saved architecture")
    for key, split in (("train_identity_sha256", inventory["train"]), ("val_identity_sha256", inventory["validation"]), ("test_identity_sha256", entry["test"])):
        if row.get(key) != split["identity_sha256"]:
            raise ValueError("Metric identity differs from physical split inventory")
    for key, members in (("train_indices", training), ("val_indices", validation)):
        indices = entry.get(key)
        if (not isinstance(indices, list) or len(indices) != len(members)
                or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices)):
            raise ValueError("Missing or duplicated training inventory indices")
    if len(model.predictions) != len(testing) or any(
        prediction.source_index not in testing
        or abs(prediction.time_s - testing[prediction.source_index][0]) > 1.1e-6
        or abs(prediction.true_km - testing[prediction.source_index][1]) > 1.1e-6
        for prediction in model.predictions
    ):
        raise ValueError("Prediction CSV differs from full-precision test inventory")
    preprocessing = model.architecture.get("input_preprocessing")
    if model.method == "RBD_SCM":
        if any(record.get("input_preprocessing") != preprocessing for record in (row, entry)):
            raise ValueError("Joint fitted preprocessing differs between sidecar/report/inventory")
    metrics = _verify_training(row, model, entry)
    best = next(epoch for epoch in model.history if epoch["epoch"] == metrics["best_epoch"])
    for split in ("train", "val"):
        for key in ("expert_load", "expert_importance", "router_entropy", "balance_loss"):
            metrics[f"best_{split}_{key}"] = best[f"{split}_{key}"]
    metrics.update(train_count=len(training), validation_count=len(validation), test_count=len(testing),
                   candidate_train_count=sampling["candidate_train_count"], parameter_count=scaling["parameter_count"],
                   active_parameter_count=scaling["active_parameter_count"], test_mae_km=model.metrics["mae_km"],
                   test_rmse_km=model.metrics["rmse_km"], input_preprocessing=preprocessing,
                   **{key: row[key] for key in ("train_identity_sha256", "val_identity_sha256", "test_identity_sha256")})
    return metrics, training, validation, testing


def load_study(study_path: Path, runs_root: Path | None = None) -> StudyResults:
    """Reject inconsistent evidence; leave every missing planned cell explicit."""
    # Lazy imports keep this module usable while dispatch constructs its registry.
    from scripts.MoE_data_scale.localization_networks.workflow.study import CHILD_STATES, child_config, selection_for, validate_plan
    from scripts.MoE_data_scale.localization_networks.workflow.tasks import model_plan
    study_path = Path(study_path).resolve()
    sources = []
    study = _read(study_path, sources)
    if (not isinstance(study, dict) or type(study.get("schema_version")) is not int
            or study["schema_version"] != 1 or study.get("pack") != PACK):
        raise ValueError("Unsupported study pack/schema")
    plan = validate_plan(study.get("plan"))
    if study.get("study_id") != plan["study_id"] or study_path.parent.name != plan["study_id"]:
        raise ValueError("Study ID differs from frozen plan/result directory")
    if digest(plan, compact=True) != study.get("plan_sha256"):
        raise ValueError("Study plan hash mismatch")
    source_files = study.get("source_files")
    if not isinstance(source_files, list) or not source_files or digest(source_files, compact=True) != study.get("source_sha256"):
        raise ValueError("Study source inventory hash mismatch")
    # Byte identities are recorded per file, never inferred from a current checkout.
    source_paths = set()
    for item in source_files:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]
                or item["path"] in source_paths or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts
                or type(item.get("size")) is not int or item["size"] < 0
                or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Invalid or duplicate study source inventory member")
        source_paths.add(item["path"])
    children = study.get("children")
    if not isinstance(children, list):
        raise ValueError("Study children must be an explicit list")
    indexed, run_ids = {}, set()
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("Invalid study child")
        seed, fraction, experts = child.get("seed"), child.get("train_fraction"), child.get("experts")
        if (type(seed) is not int or type(fraction) not in (int, float) or type(experts) is not int
                or seed not in plan["seeds"] or fraction not in plan["fractions"] or experts not in plan["experts"]):
            raise ValueError("Unexpected study seed/fraction/expert child")
        key = seed, fraction, experts
        if key in indexed or not isinstance(child.get("status"), str) or child["status"] not in CHILD_STATES:
            raise ValueError("Duplicate or invalid study child")
        config = child_config(plan, seed, fraction, experts)
        suite, selection = selection_for(plan, experts)
        jobs = model_plan(suite, selection)
        run_id = child.get("run_id")
        validate_run_identity(run_id, PACK, config, selection)
        if (run_id in run_ids or child.get("config_sha256") != digest(config)
                or child.get("network") != selection["network"] or child.get("model") != jobs[0]["model"]):
            raise ValueError("Study child configuration/selected-model/run identity mismatch")
        run_ids.add(run_id)
        indexed[key] = child
    runs_root = Path(runs_root).resolve() if runs_root is not None else study_path.parent / "children"
    rows, validations, memberships, preprocessing_states = [], {}, {}, {}
    test_members, frozen_data_identity, frozen_source_bytes, frozen_signature, lock_hash = None, None, None, None, None
    for seed in plan["seeds"]:
        for fraction in plan["fractions"]:
            for experts in plan["experts"]:
                child = indexed.get((seed, fraction, experts))
                suite, selection = selection_for(plan, experts)
                jobs = model_plan(suite, selection)
                common = {"study_id": study["study_id"], "run_id": child["run_id"] if child else None,
                          "method": plan["method"], "model": jobs[0]["model"], "network": selection["network"],
                          "seed": seed, "train_fraction": fraction, "experts": experts,
                          "run_status": child["status"] if child else "NOT_RUN"}
                if child is None or child["status"] != "PASS":
                    rows.append({**common, "status": "NOT_RUN", "reason": (child.get("error") or "child_not_pass") if child else "missing_planned_child"})
                    continue
                run_dir = runs_root / child["run_id"]
                if not run_dir.is_dir() or run_dir.is_symlink() or not run_dir.resolve().is_relative_to(runs_root.resolve()):
                    raise ValueError("PASS child result directory is missing or external")
                result = load_run(run_dir)
                expected_config = child_config(plan, seed, fraction, experts)
                if (result.config != expected_config or result.report.get("source_sha256") != study["source_sha256"]
                        or result.report.get("model_plan") != jobs or result.report.get("selection") != selection
                        or result.report.get("suite") != suite):
                    raise ValueError("Child evidence differs from frozen study request/source/model plan")
                manifest = _read(run_dir / "manifest.json", sources, run_dir)
                if manifest.get("files") != source_files:
                    raise ValueError("Child source inventory differs from frozen study source")
                lock = result.report.get("uv_lock_sha256")
                if not isinstance(lock, str) or not re.fullmatch(r"[a-f0-9]{64}", lock) or (lock_hash is not None and lock != lock_hash):
                    raise ValueError("Study child lockfile identities differ or are missing")
                lock_hash = lock
                identity = _read(run_dir / "data_identity.json", sources, run_dir)
                if (not isinstance(identity, dict) or identity.get("pack") != PACK or identity.get("schema_version") != 1
                        or digest({key: value for key, value in identity.items() if key != "cache_key"}) != identity.get("cache_key")
                        or result.report.get("dataset_cache", {}).get("cache_key") != identity["cache_key"]):
                    raise ValueError("Invalid child raw-input/generation cache identity")
                if (identity.get("dataset") != expected_config["dataset"] or identity.get("rbd") != expected_config["rbd"]
                        or identity.get("uv_lock_sha256") != lock_hash):
                    raise ValueError("Raw-input/generation identity differs from frozen configuration/lock")
                raw_inputs = identity.get("inputs")
                if (not isinstance(raw_inputs, list) or not raw_inputs
                        or any(not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]
                               or type(item.get("size")) is not int or item["size"] < 1
                               or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in raw_inputs)
                        or len({item["path"] for item in raw_inputs}) != len(raw_inputs)):
                    raise ValueError("Invalid raw-input byte inventory")
                recorded_sources = {item["path"]: item for item in source_files}
                for field in ("generator_sources", "validator_sources"):
                    items = identity.get(field)
                    if (not isinstance(items, list) or any(not isinstance(item, dict)
                            or recorded_sources.get(item.get("path")) != item for item in items)):
                        raise ValueError("Generation/validation source identity differs from frozen snapshot")
                if frozen_data_identity is not None and identity != frozen_data_identity:
                    raise ValueError("Study children used different raw inputs or generation identities")
                frozen_data_identity = identity
                sources.extend({**source, "path": str(run_dir / source["path"])} for source in result.sources)
                archive = _read(run_dir / "reports/training_inventory.json", sources, run_dir)
                if (not isinstance(archive, dict) or archive.get("pack") != PACK or archive.get("schema_version") != 1
                        or archive.get("run_id") != child["run_id"] or not isinstance(archive.get("models"), list) or len(archive["models"]) != 1):
                    raise ValueError("Invalid child training inventory report identity/model count")
                entry = archive["models"][0]
                metrics, training, validation, testing = _verify_model_evidence(result, entry, seed, fraction, experts)
                source_bytes = _source_inventory(entry.get("source_identity"))
                if frozen_source_bytes is not None and source_bytes != frozen_source_bytes:
                    raise ValueError("Study children used different source dataset bytes")
                frozen_source_bytes = source_bytes
                if seed in validations:
                    _same_members(validations[seed], validation, "Same-seed validation")
                validations[seed] = validation
                if test_members is not None:
                    _same_members(test_members, testing, "Cross-cell test")
                test_members = testing
                key = seed, fraction
                if key in memberships:
                    _same_members(memberships[key], training, "Same-seed/fraction expert training")
                memberships[key] = training
                preprocessing = result.models[0].architecture.get("input_preprocessing")
                if key in preprocessing_states and preprocessing_states[key] != preprocessing:
                    raise ValueError("Same-seed/fraction experts used different fitted block_rms state")
                preprocessing_states[key] = preprocessing
                signature = result.models[0].architecture.get("input_signature")
                if frozen_signature is not None and signature != frozen_signature:
                    raise ValueError("Study child physical input axes differ")
                frozen_signature = signature
                rows.append({**common, "status": "PASS", "reason": None, **metrics})
    for seed in plan["seeds"]:
        available = [(fraction, memberships[(seed, fraction)]) for fraction in sorted(plan["fractions"]) if (seed, fraction) in memberships]
        for (_, smaller), (_, larger) in zip(available, available[1:]):
            if not set(smaller).issubset(larger) or any(larger[key] != value for key, value in smaller.items()):
                raise ValueError("Training subsets are not physically nested across fractions")
    return StudyResults(study_path, study, rows, summarize_rows(rows, plan["seeds"]), sources)


def validation_selection(data: StudyResults) -> list[dict]:
    lookup = {(row["train_fraction"], row["experts"]): row for row in data.summaries}
    output = []
    for fraction in data.study["plan"]["fractions"]:
        candidates = [lookup[(fraction, experts)] for experts in data.study["plan"]["experts"]]
        common = {"method": data.study["plan"]["method"], "train_fraction": fraction,
                  "selection_criterion": "minimum across-seed mean best validation RMSE; exact ties use fewer experts",
                  "selection_data": "validation", "test_used_for_selection": False}
        if any(row["status"] != "PASS" for row in candidates):
            output.append({**common, "status": "NOT_RUN", "reason": "incomplete_expert_seed_matrix"})
            continue
        winner = min(candidates, key=lambda row: (row["best_val_rmse_km_mean"], row["experts"]))
        output.append({**common, "status": "PASS", "selected_experts": winner["experts"],
                       "n_seeds": winner["n_seeds"], "selected_best_val_rmse_km_mean": winner["best_val_rmse_km_mean"],
                       "selected_best_val_rmse_km_sd": winner["best_val_rmse_km_sd"],
                       "selected_test_rmse_km_mean": winner["test_rmse_km_mean"],
                       "selected_test_rmse_km_sd": winner["test_rmse_km_sd"]})
    return output


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def summarize_study(study_dir: Path) -> dict:
    """Write JSON/CSV evidence summaries after fetch, without launching work.

    Invalid evidence produces PARTIAL with a validation error and no aggregates.
    The caller can inspect status; a completed coordinator is not sufficient PASS.
    """
    study_dir = Path(study_dir).resolve()
    try:
        data = load_study(study_dir / "study.json")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report = {"schema_version": 1, "pack": PACK, "study_id": study_dir.name,
                  "status": "PARTIAL", "validation_errors": [str(exc)],
                  "cells": [], "summaries": [], "validation_selection": [], "limitations": LIMITATIONS}
        for name in ("study-cells.csv", "study-summary.csv", "validation-selection.csv"):
            _write_csv(study_dir / name, [])
    else:
        report = {"schema_version": 1, "pack": PACK, "study_id": data.study["study_id"],
                  "status": data.status, "plan_sha256": data.study["plan_sha256"], "source_sha256": data.study["source_sha256"],
                  "expected_cells": len(data.rows), "passed_cells": sum(row["status"] == "PASS" for row in data.rows),
                  "missing_cells": [row for row in data.rows if row["status"] != "PASS"], "validation_errors": [],
                  "cells": data.rows, "summaries": data.summaries, "validation_selection": validation_selection(data),
                  "sources": data.sources, "limitations": LIMITATIONS}
        for name, rows in (("study-cells.csv", report["cells"]), ("study-summary.csv", report["summaries"]),
                           ("validation-selection.csv", report["validation_selection"])):
            _write_csv(study_dir / name, rows)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(study_dir / "study-report.json", report)
    return report
