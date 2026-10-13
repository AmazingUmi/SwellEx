"""Explicit local CPU GRNN computation for one method and range_overlap direction."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.base_pack.data_processing.feature_math import segments
from scripts.base_pack.data_processing.generation_config import normalize_config
from scripts.base_pack.data_processing.generation_core import generate_method, json_value, load_inputs
from scripts.base_pack.localization_networks.common.paths import dataset_dir
from scripts.base_pack.localization_networks.common.input_contract import require_matching_input_signature
from scripts.base_pack.localization_networks.grnn.bindings import METHODS, model_name, validate_method
from scripts.base_pack.localization_networks.grnn.cli import (
    build_command, build_parser, load_artifact, predict_command, validate_arguments,
)
from scripts.base_pack.localization_networks.grnn.data import load_grnn_features
from scripts.base_pack.localization_networks.validation.datasets import classify, inspect_h5, split_check
from scripts.base_pack.localization_networks.workflow.config import load_config
from scripts.base_pack.localization_networks.workflow.remote import EXCLUDED, PACK
from scripts.base_pack.localization_networks.workflow.worker import prediction_metrics


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def python_source_paths(project: Path) -> list[Path]:
    """Capture every Python source, including dynamic loaders and package init files.

    Frozen remote projects have no Git metadata, so the inventory is taken from
    the source tree itself, using the dispatcher's generated-file exclusions.
    """
    project = Path(project).resolve()
    paths = []
    for path in sorted((project / "scripts").rglob("*.py")):
        relative = path.relative_to(project)
        if any(part in EXCLUDED or part.startswith("._") for part in relative.parts):
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                    if parent != project and project in parent.parents):
            raise ValueError(f"Symlink Python sources are not supported: {relative}")
        if path.is_file():
            paths.append(path)
    return paths


def raw_input_records(metadata: dict) -> list[dict]:
    """Verify the generator's complete raw input inventory before fitting GRNN."""
    inventory = metadata.get("input_files")
    required = {"path", "size_bytes", "mtime_ns", "sha256"}
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Dataset metadata requires a nonempty raw input_files inventory")
    elements = metadata.get("num_elements")
    event = metadata.get("effective_config", {}).get("dataset", {}).get("event_name")
    if type(elements) is not int or elements <= 0 or not isinstance(event, str):
        raise ValueError("Dataset metadata cannot determine its complete raw input inventory")
    expected_names = {"positions_vla.txt", "i9605.prn", f"SproulToVLA.{event}.txt",
                      *(f"{event}_VLA_NO_{index}.mat" for index in range(1, elements + 1))}
    if len(inventory) != len(expected_names):
        raise ValueError("Dataset metadata raw input inventory does not contain every original input")
    records, seen = [], set()
    for item in inventory:
        if (not isinstance(item, dict) or set(item) != required
                or not isinstance(item["path"], str) or not Path(item["path"]).is_absolute()
                or type(item["size_bytes"]) is not int or item["size_bytes"] < 0
                or type(item["mtime_ns"]) is not int or item["mtime_ns"] < 0
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) is None):
            raise ValueError("Dataset metadata raw input_files requires complete path/size/mtime/SHA256 records")
        path = Path(item["path"])
        if not path.is_file():
            raise ValueError(f"Dataset raw input is missing: {path}")
        record = file_record(path)
        if record["path"] in seen:
            raise ValueError(f"Dataset raw input inventory contains duplicate paths: {path}")
        seen.add(record["path"])
        if (record["size_bytes"] != item["size_bytes"]
                or path.stat().st_mtime_ns != item["mtime_ns"]
                or record["sha256"] != item["sha256"]):
            raise ValueError(f"Dataset raw input identity differs from generation metadata: {path}")
        records.append(record)
    if {Path(record["path"]).name for record in records} != expected_names:
        raise ValueError("Dataset metadata raw input inventory differs from its event and element axes")
    for field in ("range_file", "ctd_file"):
        path = metadata.get(field)
        if not isinstance(path, str) or str(Path(path).resolve()) not in seen:
            raise ValueError(f"Dataset metadata raw input inventory does not contain its {field}")
    return records


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unique(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def plot_sigma(search: list[dict], sigma: float, path: Path, side: str, method: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
        try:
            for ax, key, label in zip(axes, ("mae_km", "rmse_km", "mape_pct"),
                                     ("Mean fold MAE (km)", "Mean fold RMSE (km)", "Mean fold MAPE (%)")):
                ax.plot([row["sigma"] for row in search], [row["mean_" + key] for row in search],
                        "o-", color="#0072B2", markersize=3)
                ax.axvline(sigma, color="#444444", linestyle="--", label=f"Selected sigma: {sigma:g}")
                ax.set(xscale="log", xlabel="Gaussian kernel sigma", ylabel=label, ylim=(0, None))
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            fig.suptitle(f"{method}-GRNN range_overlap train_{side}: training-side random CV only")
            fig.savefig(path, format="pdf", facecolor="white")
        finally:
            plt.close(fig)


def run(config_path: Path, *, method: str, data: Path | None = None, root: Path = ROOT) -> dict:
    """Generate the method if needed, select sigma on train, then evaluate all test rows."""
    validate_method(method)
    model = model_name(method)
    import torch
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    config = load_config(config_path)
    if config["dataset"]["split_strategy"] != "range_overlap":
        raise ValueError("Local GRNN range_overlap computation requires split_strategy=range_overlap")
    from scripts.run_identity import make_run_id
    run_id = make_run_id("base_pack", config, {"method": method, "network": "GRNN"})
    print(f"Run: {run_id}", flush=True)
    delivery = root / "output/data" / run_id
    figures = root / "output/figure" / run_id
    delivery.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    (delivery / "logs").mkdir()
    report_path = delivery / "report.json"
    side = config["dataset"]["train_side"]
    report = {"schema_version": 1, "pack": PACK, "run_id": run_id, "status": "RUNNING",
              "method": method, "model": model,
              "selection": {"method": method, "architecture": "GRNN", "model": model},
              "purpose": "complete_range_overlap_grnn_without_accuracy_acceptance_threshold",
              "execution": {"device": "cpu", "threads": config["validation"]["threads"],
                            "python": sys.version, "platform": platform.platform(),
                            "dependencies": {name: importlib.metadata.version(name)
                                             for name in ("torch", "numpy", "scipy", "h5py", "matplotlib")}},
              "config": config, "config_file": file_record(config_path), "checks": [],
              "evaluation_scope": {"sigma_selection": "training_side_random_k_fold_only",
                                   "test_role": "final_evaluation_only",
                                   "signal_centering": "per_element_full_record_mean",
                                   "range_selection": "intersection_of_both_sides_true_range_intervals"}}
    save_json(delivery / "config.json", config)
    shutil.copyfile(config_path, delivery / "config.toml")
    sources = python_source_paths(ROOT)
    sources.extend((ROOT / "pyproject.toml", ROOT / ".python-version"))
    report["source_files"] = [file_record(path) for path in sources]
    for path in sources:
        destination = delivery / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    report["lockfile"] = file_record(ROOT / "uv.lock")
    shutil.copyfile(ROOT / "uv.lock", delivery / "source/uv.lock")
    packages = sorted(d.metadata["Name"] + "==" + d.version for d in importlib.metadata.distributions())
    (delivery / "environment.txt").write_text("\n".join(packages) + "\n")
    save_json(report_path, report)

    def phase(name, action):
        check = {"phase": name, "status": "RUNNING"}
        report["checks"].append(check)
        save_json(report_path, report)
        started = time.monotonic()
        print(f"START {name}", flush=True)
        try:
            with (delivery / "logs" / f"{name}.log").open("w") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    result = action()
            check["status"] = "PASS"
            print(f"PASS {name} ({time.monotonic() - started:.1f} s)", flush=True)
            return result
        except BaseException as exc:
            check.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            check["elapsed_s"] = time.monotonic() - started
            save_json(report_path, report)

    old_threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(config["validation"]["threads"])
        expected = normalize_config({"schema_version": 1, "dataset": config["dataset"],
                                     "rbd": config["rbd"], "result_path": f"outputs/local_grnn/{run_id}.json"}, root)
        if data is None:
            # A unique generation tag preserves existing data and failed runs.
            tag = expected["dataset"]["manual_dataset_variant_tag"]
            expected["dataset"]["manual_dataset_variant_tag"] = f"{tag + '_' if tag else ''}local_grnn_{run_id}"
            def generate():
                inputs = load_inputs(root, expected)
                d = expected["dataset"]
                base = segments(inputs["signal"], d["fs"], d["segment_duration_s"], d["segment_step_s"],
                                d["segment_start_s"], d["segment_end_s"], inputs["range_times"], inputs["ranges"])
                return generate_method(method, expected, inputs, base, root)
            generated = phase(f"generate_{method.lower()}", generate)
            directory = Path(generated["dataset_dir"])
        else:
            directory = dataset_dir(str(data)).resolve()
        train_path, test_path = (unique(directory, f"*_{split}.h5") for split in ("train", "test"))
        metadata_path = unique(directory, "*_metadata.json")
        def validate():
            # Reuse requires the complete generation contract, not just equal axes.
            metadata = json.loads(metadata_path.read_text())
            report["raw_inputs"] = raw_input_records(metadata)
            saved_config = metadata.get("effective_config", {})
            saved = saved_config.get("dataset", {})
            actual, wanted = dict(saved), dict(expected["dataset"])
            actual.pop("manual_dataset_variant_tag", None)
            wanted.pop("manual_dataset_variant_tag", None)
            if actual != wanted:
                raise ValueError(f"{method} generation configuration differs from requested range_overlap configuration")
            if method == "RBD" and saved_config.get("rbd") != expected["rbd"]:
                raise ValueError("RBD estimator configuration differs from requested generation configuration")
            if any(classify(path) != method for path in (train_path, test_path)):
                raise ValueError(f"Dataset feature method differs from requested {method}")
            records = [inspect_h5(path, method, expected)[0] for path in (train_path, test_path)]
            import numpy as np
            if any(not np.array_equal(records[0][key], records[1][key])
                   for key in ("raw_times", "raw_ranges")):
                raise ValueError(f"{method} train/test raw range track differs")
            return records, split_check(records, expected)
        records, split_validation = phase("validate_split", validate)
        report["dataset"] = {"method": method, "directory": str(directory), "train_file": str(train_path),
                             "test_file": str(test_path), "metadata_file": str(metadata_path),
                             "train_count": len(records[0]["sources"]), "test_count": len(records[1]["sources"]),
                             "split": records[0]["metadata"]["split"], "validation": split_validation}
        report["input_files"] = [file_record(path) for path in (train_path, test_path, metadata_path)]
        report["split_identity"] = {f"{split}_{key}": record[field].tolist()
                                    for split, record in zip(("train", "test"), records)
                                    for key, field in (("source_segment_idx", "sources"), ("window_center_s", "centers"))}
        model_output = delivery / "models" / method
        parser = build_parser()
        g, v = config["grnn"], config["validation"]
        build_argv = ["build", "--method", method, "--data", str(directory), "--train-data", str(train_path),
                      "--output-dir", str(model_output), "--device", "cpu", "--cv-folds", str(g["cv_folds"]),
                      "--sigma-candidates", *map(str, g["sigma_candidates"]), "--seed", str(v["seed"]),
                      "--batch-size", str(g["batch_size"])]
        args = parser.parse_args(build_argv)
        validate_arguments(args, parser)
        phase("build_reference", lambda: build_command(args))
        model_dir = model_output / model
        reference = unique(model_dir / "reference_outputs", "*_reference_*.pt")
        artifact = load_artifact(reference, torch.device("cpu"), method=method)
        train_bundle = load_grnn_features([train_path], method=method)
        require_matching_input_signature(artifact["input_signature"], train_bundle.input_signature,
                                         "GRNN reference/training data")
        if artifact["reference_sample_count"] != report["dataset"]["train_count"]:
            raise ValueError("Reference must include every training window")
        torch.testing.assert_close(artifact["x_reference"], train_bundle.x, rtol=0, atol=0)
        torch.testing.assert_close(artifact["y_reference_km"], train_bundle.y_range_km, rtol=0, atol=0)
        artifact.update(split_strategy="range_overlap", train_side=side,
                        reference_source_segment_idx=train_bundle.source_segment_idx,
                        reference_window_center_s=train_bundle.window_center_s,
                        source_h5_sha256=report["input_files"][0]["sha256"])
        torch.save(artifact, reference)
        report["reference"] = {"path": str(reference), "sigma": artifact["spread"],
                               "sigma_source": artifact["sigma_source"],
                               "reference_sample_count": artifact["reference_sample_count"]}
        report["commands"] = {"build": build_argv}
        predict_argv = ["predict", "--method", method, "--data", str(directory), "--input-data", str(test_path),
                        "--reference", str(reference), "--output-dir", str(model_output),
                        "--device", "cpu", "--batch-size", str(g["batch_size"]), "--print-first", "0",
                        "--plot-path", str(figures / "prediction.pdf")]
        report["commands"]["predict"] = predict_argv
        args = parser.parse_args(predict_argv)
        validate_arguments(args, parser)
        phase("predict_test", lambda: predict_command(args))
        csv = unique(model_dir / "test_outputs", "*_predictions_*.csv")
        expected_rows = [(int(i), float(t), float(y)) for i, t, y in
                         zip(records[1]["sources"], records[1]["centers"], records[1]["labels"])]
        metrics = phase("verify_predictions", lambda: prediction_metrics(csv, expected_rows))
        # CSV rounding is part of the delivered metrics, as in the remote workflow.
        import csv as csv_module
        with csv.open() as handle:
            rows = list(csv_module.DictReader(handle))
        metrics["mape_pct"] = sum(abs(float(row["pred_range_km"]) - float(row["true_range_km"]))
                                  / max(abs(float(row["true_range_km"])), 1e-12)
                                  for row in rows) / len(rows) * 100
        report["predictions"] = {"path": str(csv), **metrics}
        search_path = unique(model_dir / "reference_outputs", "*_sigma_search_*.json")
        report["sigma_search_path"] = str(search_path)
        phase("plot_sigma", lambda: plot_sigma(json.loads(search_path.read_text()), artifact["spread"],
                                               figures / "sigma_search.pdf", side, method))
        def verify_inputs():
            for item in [*report["source_files"], *report["input_files"], *report.get("raw_inputs", []),
                         report["config_file"], report["lockfile"]]:
                if file_record(Path(item["path"])) != item:
                    raise ValueError(f"Input/source changed during computation: {item['path']}")
        phase("verify_provenance", verify_inputs)
        report["artifacts"] = [file_record(path) for directory_ in (delivery, figures)
                               for path in sorted(directory_.rglob("*"))
                               if path.is_file() and path != report_path]
        report.update(status="PASS", elapsed_s=time.monotonic() - started)
        return report
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - started)
        raise
    finally:
        torch.set_num_threads(old_threads)
        save_json(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Existing range_overlap workflow TOML")
    parser.add_argument("--data", type=Path, help="Reuse a matching method dataset directory; otherwise generate it")
    args = parser.parse_args()
    report = run(args.config, method=args.method, data=args.data)
    print(f"PASS {report['run_id']}: {report['predictions']['samples']} test windows; "
          f"MAE {report['predictions']['mae_km']:.6f} km", flush=True)


if __name__ == "__main__":
    main()
