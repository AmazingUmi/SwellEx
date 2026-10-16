"""Verify and aggregate a complete data-size study using only the standard library.

Missing planned cells remain explicit; only PASS child runs provide observations.
No model or capacity-rule selection is made from held-out test errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

from scripts.data_scale_pack.localization_networks.workflow.config import validate_config
from scripts.data_scale_pack.localization_networks.workflow.tasks import METHODS, model_plan
from scripts.data_scale_pack.visualization.results import _json, finite, load_run
from scripts.run_identity import validate_run_identity

PACK = "data_scale_pack"
TRAINING_METRICS = (
    "best_epoch", "best_val_rmse_km", "best_val_mae_km", "best_train_loss",
    "best_train_mae_km", "best_train_rmse_km", "total_train_wall_s",
    "sum_epoch_elapsed_s", "optimizer_steps",
)
ESTIMANDS = (
    "best_val_rmse_km", "best_val_mae_km", "test_rmse_km", "test_mae_km",
    "test_bias_km", "best_train_rmse_km", "online_train_val_difference_rmse_km",
    "parameter_count", "total_train_wall_s", "sum_epoch_elapsed_s", "optimizer_steps",
)
LIMITATIONS = [
    "Capacity policy is uncalibrated; these plots do not establish optimal scaling or independent-environment generalization.",
    "Points are individual planned seeds; summaries are arithmetic mean and sample SD (ddof=1), not confidence intervals.",
    "Seed variation can change both model randomness and the training/validation partition; it is not between-environment variation.",
    "Validation identities are fixed across fractions within each seed; source windows remain temporally correlated.",
    "Test metrics evaluate the checkpoint selected by minimum validation RMSE; test metrics never select a profile or policy here.",
    "Automatic and fixed models with the same resolved profile can be duplicate computations; they stay separate and are never pooled as extra replicates.",
    "Baseline has four original blocks and different downsampling; it is distinct from the four-block medium profile.",
    "Incomplete cells remain NOT_RUN with their child status; no imputation or lines across missing fractions are used.",
    "Training RMSE is accumulated online with dropout/batch normalization in training mode while weights update; it is not a frozen-model training score or a clean generalization gap.",
    "Training cost is observed wall time in the recorded run environment, not a controlled throughput benchmark.",
]


def digest(value, *, compact=False) -> str:
    options = {"sort_keys": True, "allow_nan": False}
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def split_identity(rows) -> str:
    ordered = sorted(rows, key=lambda row: row[0])
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path: Path, sources: list[dict]) -> object:
    if not path.is_file() or path.is_symlink():
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
    histories: list[dict]
    inventories: list[dict]
    sources: list[dict]

    @property
    def status(self) -> str:
        return "PASS" if all(row["status"] == "PASS" for row in self.rows) else "PARTIAL"


def summarize_rows(rows: list[dict], seeds: list[int]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["method"], row["architecture"], row["train_fraction"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (method, architecture, fraction), group in groups.items():
        valid = [row for row in group if row["status"] == "PASS"]
        counts = {row["train_count"] for row in valid}
        if len(counts) > 1:
            raise ValueError("A method/fraction has different actual training counts across seeds")
        summary = {"method": method, "architecture": architecture, "train_fraction": fraction,
                   "status": "PASS" if len(valid) == len(seeds) else "PARTIAL" if valid else "NOT_RUN",
                   "n_seeds": len(valid), "planned_seeds": len(seeds),
                   "observed_seeds": [row["seed"] for row in valid],
                   "missing_seeds": [seed for seed in seeds if not any(row["seed"] == seed for row in valid)],
                   "train_count": next(iter(counts)) if counts else None,
                   "selected_profiles": sorted({row["selected_profile"] for row in valid})}
        for key in ESTIMANDS:
            values = [row[key] for row in valid]
            summary[key + "_mean"] = statistics.mean(values) if values else None
            summary[key + "_sd"] = statistics.stdev(values) if len(values) >= 2 else None
        summaries.append(summary)
    return summaries


def load_study(study_path: Path, runs_root: Path | None = None) -> StudyResults:
    study_path = Path(study_path).resolve()
    sources = []
    study = _read(study_path, sources)
    if (not isinstance(study, dict) or type(study.get("schema_version")) is not int or study["schema_version"] != 1 or study.get("pack") != PACK
            or not isinstance(study.get("study_id"), str) or not study["study_id"]):
        raise ValueError("Unsupported study identity/schema")
    plan = study.get("plan")
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "base_config", "fractions", "seeds", "method"} or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Study requires a complete canonical plan")
    config = validate_config(copy.deepcopy(plan["base_config"]))
    fractions, seeds = plan["fractions"], plan["seeds"]
    if (not isinstance(fractions, list) or not fractions or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1 for value in fractions)
            or len(set(fractions)) != len(fractions)):
        raise ValueError("Study fractions must be unique finite values in (0,1]")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Study seeds must be unique uint32 integers")
    method = plan["method"]
    if method is not None and method not in METHODS:
        raise ValueError("Study method must be null or a canonical single route")
    if digest(plan, compact=True) != study.get("plan_sha256"):
        raise ValueError("Study plan hash mismatch")
    if not isinstance(study.get("source_files"), list) or not study["source_files"] or digest(study["source_files"], compact=True) != study.get("source_sha256"):
        raise ValueError("Study source inventory hash mismatch")
    suite = "selected" if method else "experiment"
    selection = {"method": method, "network": None} if method else None
    jobs = model_plan(suite, selection)
    children = study.get("children")
    if not isinstance(children, list):
        raise ValueError("Study children must be an explicit list")
    indexed, run_ids = {}, set()
    for child in children:
        if (not isinstance(child, dict) or type(child.get("seed")) is not int
                or type(child.get("train_fraction")) not in (int, float) or not isinstance(child.get("status"), str)):
            raise ValueError("Invalid study child")
        key = (child.get("seed"), child.get("train_fraction"))
        if key[0] not in seeds or key[1] not in fractions or key in indexed:
            raise ValueError("Unexpected or duplicate study seed/fraction child")
        run_id = child.get("run_id")
        expected_config = copy.deepcopy(config)
        expected_config["validation"]["seed"], expected_config["training"]["train_fraction"] = key
        validate_run_identity(run_id, PACK, expected_config, selection)
        if run_id in run_ids or child.get("config_sha256") != digest(expected_config):
            raise ValueError("Duplicate child run or configuration hash mismatch")
        run_ids.add(run_id)
        indexed[key] = child
    runs_root = Path(runs_root).resolve() if runs_root is not None else study_path.parent / "children"
    rows, histories, inventories = [], [], []
    validations, tests, populations, memberships = {}, {}, {}, {}
    uv_lock = None
    frozen_data_identity = None
    for seed in seeds:
        for fraction in fractions:
            child = indexed.get((seed, fraction))
            common = {"study_id": study["study_id"], "run_id": child["run_id"] if child else None,
                      "seed": seed, "train_fraction": fraction, "run_status": child.get("status", "UNKNOWN") if child else "NOT_RUN"}
            if child is None or child.get("status") != "PASS":
                reason = child.get("error") or "child_not_pass" if child else "missing_planned_child"
                rows.extend({**common, **job, "status": "NOT_RUN", "reason": reason} for job in jobs)
                continue
            run_dir = runs_root / child["run_id"]
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise ValueError(f"PASS child result directory is unavailable: {run_dir}")
            result = load_run(run_dir)
            expected_config = copy.deepcopy(config)
            expected_config["validation"]["seed"] = seed
            expected_config["training"]["train_fraction"] = fraction
            if (result.config != expected_config or result.report.get("source_sha256") != study["source_sha256"]
                    or result.report.get("model_plan") != jobs):
                raise ValueError("Child evidence differs from frozen study configuration/source/model plan")
            lock = result.report.get("uv_lock_sha256")
            if not isinstance(lock, str) or len(lock) != 64 or (uv_lock is not None and lock != uv_lock):
                raise ValueError("Study child lockfile identities differ or are missing")
            uv_lock = lock
            identity = _read(run_dir / "data_identity.json", sources)
            if (not isinstance(identity, dict) or identity.get("pack") != PACK or identity.get("schema_version") != 1
                    or digest({key: value for key, value in identity.items() if key != "cache_key"}) != identity.get("cache_key")
                    or result.report.get("dataset_cache", {}).get("cache_key") != identity["cache_key"]):
                raise ValueError("Child raw-input/generation cache identity is invalid")
            if frozen_data_identity is not None and identity != frozen_data_identity:
                raise ValueError("Study children used different raw inputs or feature-generation identities")
            frozen_data_identity = identity
            sources.extend({**source, "path": str(run_dir / source["path"])} for source in result.sources)
            archive = _read(run_dir / "reports/training_inventory.json", sources)
            if (not isinstance(archive, dict) or archive.get("schema_version") != 1 or archive.get("pack") != PACK
                    or archive.get("run_id") != child["run_id"] or not isinstance(archive.get("models"), list)):
                raise ValueError("Invalid training inventory report identity")
            entries = archive["models"]
            entry_map = {(entry.get("method"), entry.get("model")): entry for entry in entries if isinstance(entry, dict)}
            if len(entry_map) != len(entries) or set(entry_map) != {(job["method"], job["model"]) for job in jobs}:
                raise ValueError("Training inventory does not cover the complete model plan")
            metric_map = {(row["method"], row["model"]): row for row in result.report["metrics"]}
            for job, model in zip(jobs, result.models, strict=True):
                if (job["method"], job["model"]) != (model.method, model.model):
                    raise ValueError("Child model ordering differs from canonical plan")
                row, entry = metric_map[(model.method, model.model)], entry_map[(model.method, model.model)]
                if entry.get("seed") != seed or entry.get("train_fraction") != fraction or row.get("training_inventory_path") != "reports/training_inventory.json":
                    raise ValueError("Model training inventory seed/fraction/path mismatch")
                if len(model.history) != config["training"]["epochs"]:
                    raise ValueError("Child epoch count differs from the frozen study budget")
                inventory = entry.get("training_inventory")
                if not isinstance(inventory, dict) or inventory != model.architecture.get("training_inventory") or inventory.get("schema_version") != 1:
                    raise ValueError("Sidecar and training inventory report disagree")
                training = verify_split(inventory.get("train"), "training")
                validation = verify_split(inventory.get("validation"), "validation")
                testing = verify_split(entry.get("test"), "test")
                if set(training) & (set(validation) | set(testing)) or set(validation) & set(testing):
                    raise ValueError("Training/validation/test physical sample identities overlap")
                scaling, sampling = model.architecture["scaling"], model.architecture["sampling"]
                if (len(training) != scaling["train_count"] or len(validation) != sampling["validation_count"]
                        or len(training) != math.floor(sampling["candidate_train_count"] * fraction)
                        or entry.get("sampling") != sampling):
                    raise ValueError("Actual member count differs from the recorded training budget")
                if row.get("parameter_count") != scaling["parameter_count"]:
                    raise ValueError("Reported training parameter count differs from saved architecture")
                expected_steps = math.ceil(len(training) / config["training"]["batch_size"])
                if any(epoch.get("optimizer_steps_epoch") != expected_steps for epoch in model.history):
                    raise ValueError("Optimizer steps differ from actual training count/batch geometry")
                coverage_metrics = {}
                for split_name, members, actual in (("validation", validation, inventory.get("validation_coverage")), ("test", testing, entry.get("test_coverage"))):
                    lower, upper = min(value[1] for value in training.values()), max(value[1] for value in training.values())
                    count = sum(lower <= value[1] <= upper for value in members.values())
                    expected = {"within_train_range_count": count, "within_train_range_fraction": count / len(members)}
                    if actual != expected:
                        raise ValueError("Recorded held-out range coverage differs from split membership")
                    coverage_metrics[split_name + "_within_train_range_fraction"] = expected["within_train_range_fraction"]
                for field, split in (("train_identity_sha256", inventory["train"]), ("val_identity_sha256", inventory["validation"]), ("test_identity_sha256", entry["test"])):
                    if row.get(field) != split["identity_sha256"]:
                        raise ValueError("Metric identity hash differs from physical split inventory")
                for index_field, members in (("train_indices", training), ("val_indices", validation)):
                    indices = entry.get(index_field)
                    if not isinstance(indices, list) or len(indices) != len(members) or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
                        raise ValueError("Training inventory indices are missing or duplicated")
                if len(model.predictions) != len(testing) or any(
                    prediction.source_index not in testing or abs(prediction.time_s - testing[prediction.source_index][0]) > 1.1e-6
                    or abs(prediction.true_km - testing[prediction.source_index][1]) > 1.1e-6 for prediction in model.predictions
                ):
                    raise ValueError("Prediction CSV differs from full-precision test inventory")
                validation_key = (model.method, seed)
                if validation_key in validations:
                    _same_members(validations[validation_key], validation, "Same-seed validation")
                validations[validation_key] = validation
                if model.method in tests:
                    _same_members(tests[model.method], testing, "Cross-seed test")
                tests[model.method] = testing
                membership_key = (model.method, seed, fraction)
                if membership_key in memberships:
                    _same_members(memberships[membership_key], training, "Same-cell model training")
                memberships[membership_key] = training
                sources_identity = entry.get("source_identity")
                if not isinstance(sources_identity, dict) or sources_identity.get("version") != 1:
                    raise ValueError("Training source byte identity is missing")
                normalized = {}
                for field in ("train_sources", "validation_sources"):
                    items = sources_identity.get(field)
                    if not isinstance(items, list) or not items:
                        raise ValueError("Training source inventory is missing")
                    if any(not isinstance(item, dict) or type(item.get("size_bytes")) is not int or item["size_bytes"] < 1
                           or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in items):
                        raise ValueError("Training source byte inventory has invalid size/hash")
                    normalized[field] = [{key: item[key] for key in ("size_bytes", "sha256")} for item in items]
                if model.method in populations and normalized != populations[model.method]:
                    raise ValueError("Study children used different source dataset bytes")
                populations[model.method] = normalized
                training_metrics = _verify_training(row, model, entry)
                observation = {**common, **job, "status": "PASS", "reason": None,
                               **training_metrics, **coverage_metrics, "train_count": len(training), "validation_count": len(validation),
                               "test_count": len(testing), "candidate_train_count": sampling["candidate_train_count"],
                               "selection_mode": scaling["selection_mode"], "selected_profile": scaling["selected_profile"],
                               "blocks": sum(scaling["stage_repeats"]), "parameter_count": scaling["parameter_count"],
                               "parameter_budget": scaling["parameter_budget"], "test_rmse_km": model.metrics["rmse_km"],
                               "test_mae_km": model.metrics["mae_km"], "test_bias_km": model.metrics["bias_km"],
                               "online_train_val_difference_rmse_km": row["best_val_rmse_km"] - row["best_train_rmse_km"],
                               "matched_fixed_architecture": "complex_CNN_" + scaling["selected_profile"] if scaling["selection_mode"] == "auto" else None,
                               **{key: row[key] for key in ("train_identity_sha256", "val_identity_sha256", "test_identity_sha256")}}
                rows.append(observation)
                histories.extend({**common, **job, "train_count": len(training), **epoch} for epoch in model.history)
                inventories.append({**common, **entry})
    for method in METHODS:
        for seed in seeds:
            available = [(fraction, memberships[(method, seed, fraction)]) for fraction in sorted(fractions) if (method, seed, fraction) in memberships]
            for (_, smaller), (_, larger) in zip(available, available[1:]):
                if not set(smaller).issubset(larger) or any(larger[key] != value for key, value in smaller.items()):
                    raise ValueError("Training subsets are not physically nested across fractions")
    for left, right in zip(list(tests.values()), list(tests.values())[1:]):
        _same_members(left, right, "Cross-method test")
    return StudyResults(study_path, study, rows, summarize_rows(rows, seeds), histories, inventories, sources)


def validation_selection(data: StudyResults) -> list[dict]:
    """Describe validation-selected fixed profiles without tuning the auto rule."""
    profiles = ("complex_CNN_small", "complex_CNN_medium", "complex_CNN_large", "complex_CNN_xlarge")
    lookup = {(row["method"], row["train_fraction"], row["architecture"]): row for row in data.summaries}
    observations = {(row["method"], row["train_fraction"], row["architecture"], row["seed"]): row
                    for row in data.rows if row["status"] == "PASS"}
    methods = (data.study["plan"]["method"],) if data.study["plan"]["method"] else METHODS
    results = []
    for method in methods:
        for fraction in data.study["plan"]["fractions"]:
            candidates = [lookup[(method, fraction, architecture)] for architecture in profiles]
            common = {"method": method, "train_fraction": fraction,
                      "selection_criterion": "minimum across-seed mean best validation RMSE; exact ties use fewer parameters",
                      "scope": "descriptive development-validation selection, not a calibrated automatic policy"}
            if any(row["status"] != "PASS" for row in candidates):
                results.append({**common, "status": "NOT_RUN", "reason": "incomplete_fixed_profile_seed_matrix"})
                continue
            winner = min(candidates, key=lambda row: (row["best_val_rmse_km_mean"], row["parameter_count_mean"]))
            selected = {**common, "status": "PASS", "selected_architecture": winner["architecture"],
                        "train_count": winner["train_count"], "n_seeds": winner["n_seeds"],
                        "parameter_count": winner["parameter_count_mean"]}
            for field in ("best_val_rmse_km", "test_rmse_km"):
                for stat in ("mean", "sd"):
                    selected["selected_" + field + "_" + stat] = winner[field + "_" + stat]
            for reference in ("complex_CNN", "complex_CNN_baseline"):
                summary = lookup[(method, fraction, reference)]
                prefix = "auto" if reference == "complex_CNN" else "baseline"
                selected[prefix + "_status"] = summary["status"]
                for field in ("best_val_rmse_km", "test_rmse_km"):
                    for stat in ("mean", "sd"):
                        selected[prefix + "_" + field + "_" + stat] = summary[field + "_" + stat]
                if reference == "complex_CNN" and summary["status"] == "PASS":
                    for field in ("best_val_rmse_km", "test_rmse_km"):
                        differences = [observations[(method, fraction, reference, seed)][field]
                                       - observations[(method, fraction, winner["architecture"], seed)][field]
                                       for seed in data.study["plan"]["seeds"]]
                        selected["auto_minus_selected_" + field + "_mean"] = statistics.mean(differences)
                        selected["auto_minus_selected_" + field + "_sd"] = statistics.stdev(differences) if len(differences) > 1 else None
            results.append(selected)
    return results
