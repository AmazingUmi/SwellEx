"""Merge existing result evidence with the standard library; never run models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.localization_networks.grnn.bindings import model_name
from scripts.localization_networks.workflow.tasks import validate_selection
from scripts.visualization.results import LEGACY_TASKS_SHA256, TASKS_SOURCE, is_grnn, load_run

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path, sources: dict) -> object:
    remember(path, sources)
    return json.loads(path.read_text(encoding="utf-8"))


def remember(path: Path, sources: dict) -> dict:
    path = path.resolve()
    item = {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest(path.read_bytes())}
    sources[str(path)] = item
    return item


def one_file(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def selected_cv_row(search: list[dict], sigma: float) -> dict:
    selected = min(search, key=lambda row: row["mean_mape_pct"])
    if sigma != selected["sigma"]:
        raise ValueError("Selected sigma differs from training CV mean MAPE optimum")
    return selected


def prediction_metrics(path: Path, reported: dict, sources: dict) -> dict:
    info = remember(path, sources)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != reported["samples"]:
        raise ValueError(f"Prediction count differs from report: {path}")
    identity, errors, labels = [], [], []
    for index, row in enumerate(rows):
        truth, pred, time = (float(row[k]) for k in
                             ("true_range_km", "pred_range_km", "window_center_s"))
        if int(row["sample"]) != index or not all(map(math.isfinite, (truth, pred, time))) or truth <= 0:
            raise ValueError(f"Invalid prediction row {index}: {path}")
        identity.append([int(row["source_segment_idx"]), time, truth])
        labels.append(truth)
        errors.append(pred - truth)
    if len({row[0] for row in identity}) != len(rows):
        raise ValueError(f"Duplicate prediction source indices: {path}")
    n = len(rows)
    absolute = sorted(map(abs, errors))
    mae, rmse, maximum = sum(absolute) / n, math.sqrt(sum(e * e for e in errors) / n), absolute[-1]
    for name, computed in (("mae_km", mae), ("rmse_km", rmse), ("max_error_km", maximum)):
        if name not in reported or not math.isclose(computed, reported[name], rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"{name} differs from report: {path}")
    if reported.get("source_segment_indices") != [row[0] for row in identity]:
        raise ValueError(f"Prediction source indices differ from report: {path}")
    position = (n - 1) * .95
    lo = int(position)
    p95 = absolute[lo] + (absolute[min(lo + 1, n - 1)] - absolute[lo]) * (position - lo)
    return {"samples": n, "mae_m": mae * 1000, "rmse_m": rmse * 1000,
            "mape_pct": sum(abs(e) / y for e, y in zip(errors, labels)) / n * 100,
            "bias_m": sum(errors) / n * 1000, "p95_abs_m": p95 * 1000,
            "max_error_m": maximum * 1000, "true_range_min_km": min(labels),
            "true_range_max_km": max(labels), "test_time_min_s": min(row[1] for row in identity),
            "test_time_max_s": max(row[1] for row in identity), "csv": info,
            "test_identity_sha256": digest(json.dumps(identity, separators=(",", ":")).encode())}


def run_records(run: Path, sources: dict) -> list[dict]:
    report = read_json(run / "report.json", sources)
    if not isinstance(report, dict):
        raise ValueError(f"Invalid recorded result report: {run}")
    if report.get("status") != "PASS" or not ("metrics" in report or "predictions" in report):
        return []
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1 or report.get("run_id") != run.name:
        raise ValueError(f"Recorded result schema/run identity differs: {run}")
    local = "predictions" in report
    if local and report.get("purpose") != "complete_range_overlap_grnn_without_accuracy_acceptance_threshold":
        raise ValueError(f"Local GRNN result purpose differs: {run}")
    historical = False
    if not local:
        verified = load_run(run)
        for item in verified.sources:
            remember(run / item["path"], sources)
        historical = "model_plan" not in report
    request = report if local else read_json(run / "request.json", sources)
    config = request["config"]
    dataset = config["dataset"]
    original = dataset["split_strategy"]
    analysis = "full_interleaved" if original == "periodic" and dataset["train_test_ratio"] == [4, 1] else original
    if analysis not in ("range_overlap", "full_interleaved"):
        raise ValueError(f"Unsupported historical split: {original}")
    raw = report["raw_inputs"] if local else read_json(run / "data_identity.json", sources)["inputs"]
    snapshot = []
    if local and (run / "source").is_dir():
        for original_file in [*report["source_files"], report["lockfile"]]:
            relative = Path(original_file["path"]).relative_to(PROJECT)
            saved = remember(run / "source" / relative, sources)
            if saved["sha256"] != original_file["sha256"] or saved["size_bytes"] != original_file["size_bytes"]:
                raise ValueError(f"Historical source snapshot differs from saved inventory: {saved['path']}")
            snapshot.append(saved)
    if not local:
        read_json(run / "manifest.json", sources)
    normalized_raw = [{"path": item["path"].split("Origindata/")[-1], "sha256": item["sha256"]} for item in raw]
    raw_hash = digest(json.dumps(sorted(normalized_raw, key=lambda x: x["path"]), sort_keys=True).encode())
    if local:
        identity_fields = {"method", "model", "selection"}
        if identity_fields & set(report):
            if not identity_fields <= set(report):
                raise ValueError("Local GRNN report requires method, model and selection")
            selection = validate_selection(report["selection"])
            if (selection["architecture"] != "GRNN" or report["method"] != selection["method"]
                    or report["model"] != selection["model"]):
                raise ValueError("Local GRNN method/model differs from its selection")
        else:
            tasks = [item for item in report.get("source_files", [])
                     if str(item.get("path", "")).endswith("/" + TASKS_SOURCE)]
            if len(tasks) != 1 or tasks[0].get("sha256") != LEGACY_TASKS_SHA256:
                raise ValueError("Missing local GRNN method without the known historical source contract")
            selection = {"method": "SCM", "architecture": "GRNN", "model": model_name("SCM")}
            historical = True
        metrics = [{**report["predictions"], "method": selection["method"], "model": selection["model"]}]
    else:
        metrics = report["metrics"]
    records = []
    for metric in metrics:
        method, model = metric["method"], metric["model"]
        grnn = is_grnn(method, model)
        model_dir = run / "project/outputs/networks_results" / method / model
        csv_path = Path(metric["path"]) if local else one_file(model_dir / "test_outputs", "*predictions*.csv")
        record = {"run_id": report["run_id"], "method": method, "model": model,
                  "original_split": original, "analysis_split": analysis,
                  "train_side": dataset["train_side"] if analysis == "range_overlap" else None,
                  "evidence": "csv_verified", "execution": report.get("execution", config["validation"]),
                  "dataset_config": dataset, "raw_inputs": raw, "raw_input_identity_sha256": raw_hash,
                  "source_report": remember(run / "report.json", sources),
                  "source_sha256": report.get("source_sha256"),
                  "source_inventory": report.get("source_files", []),
                  "historical_source_snapshot": snapshot,
                  "source_verification": "saved_snapshot_hashes_verified" if snapshot else "recorded_inventory_only",
                  "configured_neural_epochs": config["training"]["epochs"],
                  "grnn_computation_config": {"dataset": {k: v for k, v in dataset.items() if k != "manual_dataset_variant_tag"},
                                              "grnn": config["grnn"], "seed": config["validation"]["seed"]},
                  "validation_evidence": report.get("validation_evidence"),
                  "model_plan_contract": "historical_scm_only" if historical else "explicit_current",
                  **prediction_metrics(csv_path, metric, sources)}
        if grnn:
            search_path = Path(report["sigma_search_path"]) if local else one_file(model_dir / "reference_outputs", "*sigma_search*.json")
            search = read_json(search_path, sources)
            if local:
                sigma, count = report["reference"]["sigma"], report["reference"]["reference_sample_count"]
                record["evaluation_scope"] = report["evaluation_scope"]
                metadata = read_json(Path(report["dataset"]["metadata_file"]), sources)
                record["split_metadata"] = metadata["split"]
            else:
                build_log = one_file(run / "logs", "grnn_build.log" if historical else f"grnn_{method}_build.log")
                remember(build_log, sources)
                sigma_line = next(line for line in build_log.read_text().splitlines() if line.startswith("Spread sigma:"))
                sigma = float(sigma_line.split()[2])
                contract = next(c["result"] for c in report["checks"] if c["name"] == "dataset_contract")
                train = [c["details"]["valid_samples"] for c in contract["checks"]
                         if c.get("details", {}).get("method") == method and c["details"].get("split") == "train"]
                if len(train) != 1:
                    raise ValueError(f"Ambiguous GRNN reference count: {run}")
                count = train[0]
            selected = selected_cv_row(search, sigma)
            record.update(epochs=None, reference_samples=count, sigma=sigma,
                          cv_selection="minimum_training_random_k_fold_mean_mape_pct",
                          cv_folds=config["grnn"]["cv_folds"], cv_candidates=len(search),
                          cv_selected=selected, cv_search=remember(search_path, sources))
        else:
            history_path = one_file(model_dir / "train_outputs", "*history*.json")
            history = read_json(history_path, sources)
            epochs = metric["epochs"]
            if [row["epoch"] for row in history] != list(range(1, epochs + 1)):
                raise ValueError(f"Training history does not confirm {epochs} epochs: {history_path}")
            record.update(epochs=epochs, history=remember(history_path, sources))
        records.append(record)
    if len({r["test_identity_sha256"] for r in records}) != 1:
        raise ValueError(f"Models in one run have different test identities: {run}")
    return records


def merge(input_dir: Path) -> tuple[dict, dict]:
    sources, records, summaries, diagnostics = {}, [], [], []
    for run in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if (run / "report.json").is_file():
            records.extend(run_records(run, sources))
        if (run / "diagnostics.json").is_file():
            diagnostic = read_json(run / "diagnostics.json", sources)
            source = remember(run / "diagnostics.json", sources)
            diagnostics.append({"source": source, "source_run": diagnostic["source_run"],
                                "purpose": diagnostic["purpose"], "source_script_sha256": diagnostic.get("source_script_sha256"),
                                "constant_baseline": diagnostic.get("constant_baseline"),
                                "grnn_nearest": diagnostic.get("grnn_nearest")})
            for split in diagnostic.get("other_splits", []):
                for metric in split["metrics"]:
                    grnn = is_grnn(metric["method"], metric["model"])
                    summaries.append({"run_id": split["run"], "analysis_split": split["split"],
                                      "train_side": split["side"] if split["split"] == "range_overlap" else None,
                                      "evidence": "summary_only", "actual_epochs": None if grnn else split["neural_epochs"],
                                      "epochs_evidence": "not_applicable" if grnn else "diagnostic_summary_only",
                                      "source": source, "method": metric["method"], "model": metric["model"],
                                      "samples": metric["samples"], "mae_m": metric["mae_km"] * 1000,
                                      "rmse_m": metric["rmse_km"] * 1000})
    canonical, duplicates = [], []
    ordered = sorted(records, key=lambda r: (bool(r["validation_evidence"]),
                     r["configured_neural_epochs"],
                     r["run_id"]), reverse=True)
    keys = {}
    for record in ordered:
        key = (record["analysis_split"], record["train_side"], record["method"], record["model"], record["csv"]["sha256"],
               record["raw_input_identity_sha256"], json.dumps(record["grnn_computation_config"], sort_keys=True))
        if is_grnn(record["method"], record["model"]) and key in keys:
            duplicates.append({"duplicate": record, "canonical_run_id": keys[key],
                               "reason": "same_inputs_and_config_identical_csv_recomputation_consistency",
                               "independent_repeat": False})
        else:
            canonical.append(record)
            keys[key] = record["run_id"]
    for summary in summaries:
        matches = [r for r in canonical if r["method"] == summary["method"] and r["model"] == summary["model"] and
                   r["analysis_split"] == summary["analysis_split"] and r["train_side"] == summary["train_side"]]
        summary["available_csv_matches"] = [{"run_id": r["run_id"], "mae_delta_m": r["mae_m"] - summary["mae_m"],
                                            "rmse_delta_m": r["rmse_m"] - summary["rmse_m"]} for r in matches]
    return ({"schema_version": 1, "status": "PASS", "purpose": "existing_result_comparison_without_model_execution",
             "created_at": datetime.now(timezone.utc).isoformat(), "input_dir": str(input_dir),
             "historical_split_mapping": {"periodic_4_1": "full_interleaved; analysis only, no execution alias"},
             "records": sorted(canonical, key=lambda r: (r["analysis_split"], r["run_id"], r["method"], r["model"])),
             "duplicates": duplicates, "summary_only": summaries, "diagnostics": diagnostics},
            {"schema_version": 1, "files": sorted(sources.values(), key=lambda x: x["path"])})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "output/data")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    comparison, provenance = merge(args.input_dir.resolve())
    target = args.output_dir or PROJECT / "output/data" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-comparison-" + uuid4().hex[:8])
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    provenance["generator"] = remember(Path(__file__), {})
    for name, value in (("comparison.json", comparison), ("provenance.json", provenance)):
        (target / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["run_id", "method", "model", "original_split", "analysis_split", "train_side", "evidence", "epochs",
              "reference_samples", "samples", "sigma", "mae_m", "rmse_m", "mape_pct", "bias_m", "p95_abs_m",
              "max_error_m", "true_range_min_km", "true_range_max_km", "test_identity_sha256"]
    with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison["records"])
    print(json.dumps({"status": "PASS", "output_dir": str(target), "records": len(comparison["records"]),
                      "duplicates": len(comparison["duplicates"]), "summary_only": len(comparison["summary_only"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
