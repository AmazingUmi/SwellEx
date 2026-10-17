"""Explicit CUDA demo entry; quick uses synthetic fixtures exclusively."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import traceback
import unittest


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _quick_checks(device):
    os.environ["SWELLEX_MOE_LEARN_TEST_DEVICE"] = str(device)
    root = Path(__file__).resolve().parents[3]
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), pattern="test_*.py", top_level_dir=str(root))
    def cases(items):
        for item in items:
            if isinstance(item, unittest.TestSuite):
                yield from cases(item)
            else:
                yield item
    records = []
    for case in cases(suite):
        print(f"CUDA contract: {case.id()}", flush=True)
        result = unittest.TestResult()
        case.run(result)
        passed = result.wasSuccessful() and not result.skipped and result.testsRun == 1
        records.append({"name": case.id(), "status": "PASS" if passed else "FAIL", "tests_run": result.testsRun,
                        "errors": [message for _, message in result.errors + result.failures], "skipped": result.skipped})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("quick", "smoke", "experiment"), required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"pack": "MoE_learn", "suite": args.suite, "status": "FAIL", "scientific_performance_evidence": False}
    try:
        from .common.training import require_cuda, run_stages
        from .common.checkpoints import load_checkpoint
        from .workflow.config import validate_config
        import torch
        requested = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
        effective = deepcopy(requested)
        device = require_cuda(effective["validation"]["device"])
        threads = effective["validation"]["threads"]
        if type(threads) is not int or threads <= 0:
            raise ValueError("validation.threads must be a positive integer")
        torch.set_num_threads(threads)
        report.update({"device": str(device), "torch_version": str(torch.__version__),
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "gpu": torch.cuda.get_device_name(device), "requested_config": requested})
        if args.suite == "quick":
            from .tests.synthetic import synthetic_data
            training = effective["training"]
            if (any(training[key] != 1 for key in ("initial_epochs", "adapt_epochs", "warmup_epochs"))
                    or training["base_channels"] > 4 or training["batch_size"] > 4 or effective["continual"]["stages"] != 2):
                raise ValueError("Quick requires the bounded workflow-resolved configuration: two stages, one epoch, c<=4, batch<=4")
            report.update({"scope": "synthetic_cuda_contracts_and_two_stage_demo", "input_kind": "synthetic",
                           "swellex_training": False})
            _json(output / "resolved-config.json", effective)
            report["checks"] = _quick_checks(device)
            if not report["checks"] or any(check["status"] != "PASS" for check in report["checks"]):
                raise RuntimeError("One or more synthetic CUDA contracts failed")
            prepared = synthetic_data(effective["training"]["seed"], effective["continual"]["stages"])
        else:
            from .data import load_demo_data
            report.update({"scope": "real_data_continual_demo", "input_kind": "SwellEx", "swellex_training": True})
            _json(output / "resolved-config.json", effective)
            prepared = load_demo_data(effective, output)
        result = run_stages(effective, prepared, output, device=device)
        for row in result["stages"]:
            _, saved = load_checkpoint(output / row["checkpoint"], device, expected_metadata=prepared["metadata"])
            if saved["stage"] != row["stage"] or saved["model_state_sha256"] != row["deployed_state_sha256"]:
                raise RuntimeError("Stage checkpoint differs from its evaluated deployment")
            row["checkpoint_restore"] = "PASS"
        _, checkpoint = load_checkpoint(output / "checkpoint.pt", device, expected_metadata=prepared["metadata"])
        report.update(result)
        report.update({"status": "PASS", "effective_config": effective, "checkpoint_restore": "PASS",
                       "checkpoint_stage": checkpoint["stage"], "checkpoint": "checkpoint.pt",
                       "scope_note": "Engineering demonstration only; no calibrated controller or continual-learning superiority claim."})
        with (output / "stages.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "action", "experts", "new_training_rows", "cumulative_training_rows", "cumulative_after_training_rows", "current_rmse_km", "historic_rmse_km", "test_mae_km", "test_rmse_km", "replay_rows", "novelty"])
            writer.writeheader()
            for row in result["stages"]:
                writer.writerow({"stage": row["stage"], "action": row["action"], "experts": row["experts"],
                                 "new_training_rows": row["new_training_rows"],
                                 "cumulative_training_rows": row["cumulative_training_rows"],
                                 "cumulative_after_training_rows": row["cumulative_after_training_rows"],
                                 "current_rmse_km": row["current"]["rmse_km"],
                                 "historic_rmse_km": None if row["historic"] is None else row["historic"]["rmse_km"],
                                 "test_mae_km": None if row["test"] is None else row["test"]["mae_km"],
                                 "test_rmse_km": None if row["test"] is None else row["test"]["rmse_km"],
                                 "replay_rows": row["replay_rows"], "novelty": row.get("novelty")})
        if args.suite != "quick":
            from scripts.MoE_learn.visualization.plot_stages import plot_stages
            # Plotting reads the finalized report; it cannot influence training.
            _json(output / "demo-report.json", report)
            plot_stages(output / "demo-report.json", output / "figures")
    except Exception as error:
        report.update({"status": "FAIL", "error": str(error), "traceback": traceback.format_exc()})
        print(report["traceback"], flush=True)
    _json(output / "demo-report.json", report)
    print(f"MoE_learn {args.suite}: {report['status']}; report={output / 'demo-report.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
