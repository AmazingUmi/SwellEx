"""Standard-library checks for read-only historical evidence aggregation."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.localization_networks.workflow import compare_results as compare
from scripts.localization_networks.workflow.tasks import model_plan, resolve_selection
from scripts.visualization.results import PREDICTION_FIELDS, read_predictions, summarize


class CompareResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def csv_fixture(self, path, source_indices=(1, 2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = PREDICTION_FIELDS
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, source in enumerate(source_indices):
                writer.writerow(dict(zip(fields, [i, source, i + .5, 1.1 + i, 1 + i, .1])))
        return {"samples": 2, "source_segment_indices": [1, 2],
                "mae_km": .1, "rmse_km": .1, "max_error_km": .1}

    def test_modified_predictions_and_source_identity_are_rejected(self):
        path = self.root / "prediction.csv"
        reported = self.csv_fixture(path)
        self.assertAlmostEqual(compare.prediction_metrics(path, reported, {})["mae_m"], 100)
        path.write_text(path.read_text().replace("1.1", "1.4"))
        with self.assertRaisesRegex(ValueError, "mae_km differs"):
            compare.prediction_metrics(path, reported, {})
        self.csv_fixture(path, source_indices=(3, 4))
        with self.assertRaisesRegex(ValueError, "source indices differ"):
            compare.prediction_metrics(path, reported, {})

    def remote_fixture(self):
        run = self.root / "run"
        run.mkdir()
        dataset = {"split_strategy": "full_interleaved", "train_test_ratio": [4, 1]}
        config = {"dataset": dataset, "validation": {"seed": 2026},
                  "training": {"epochs": 1}, "grnn": {"cv_folds": 5}}
        metrics = []
        plan = model_plan("smoke")
        checks = []
        for item in plan:
            method, model = item["method"], item["model"]
            model_dir = run / "project/outputs/networks_results" / method / model
            path = model_dir / "test_outputs/example_predictions_1012_120000.csv"
            self.csv_fixture(path)
            metric = {"method": method, "model": model, **summarize(read_predictions(path.read_bytes()))}
            if item["architecture"] == "GRNN":
                sigma = {"RBD": .1, "ELM": .2, "SCM": .3}[method]
                search = model_dir / "reference_outputs/example_sigma_search_1012_120000.json"
                search.parent.mkdir()
                search.write_text(json.dumps([{"sigma": sigma, "mean_mape_pct": .1,
                                                "mean_mae_km": .001, "mean_rmse_km": .002}]))
                (run / "logs").mkdir(exist_ok=True)
                (run / "logs" / f"grnn_{method}_build.log").write_text(f"Spread sigma: {sigma}\n")
            else:
                history = model_dir / "train_outputs/example_history_1012_120000.json"
                history.parent.mkdir()
                history.write_text(json.dumps([{"epoch": 1, "lr": .01, "loss_space": "km",
                                                "train_loss": .2, "val_loss": .3,
                                                "train_mae_km": .2, "val_mae_km": .3,
                                                "train_rmse_km": .3, "val_rmse_km": .4}]))
                metric["epochs"] = 1
            metrics.append(metric)
            checks.append({"name": f"evaluate_{method}_{model}", "status": "PASS"})
        checks.append({"name": "dataset_contract", "status": "PASS", "result": {"checks": [
            {"details": {"method": method, "split": "train", "valid_samples": count}}
            for method, count in (("RBD", 7), ("ELM", 8), ("SCM", 9))]}})
        for name, data in (("report.json", {"schema_version": 1, "status": "PASS", "run_id": "run",
                                           "suite": "smoke", "purpose": "engineering_validation",
                                           "model_plan": plan, "checks": checks, "metrics": metrics}),
                           ("request.json", {"config": config}), ("data_identity.json", {"inputs": []}),
                           ("manifest.json", {})):
            (run / name).write_text(json.dumps(data))
        return run

    def test_different_test_windows_between_models_are_rejected(self):
        run = self.remote_fixture()
        path = run / "project/outputs/networks_results/ELM/elm_complex_cnn_range/test_outputs/example_predictions_1012_120000.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["window_center_s"] = float(row["window_center_s"]) + 10
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        report = json.loads((run / "report.json").read_text())
        report["metrics"][1].update(summarize(read_predictions(path.read_bytes())))
        (run / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "identical test"):
            compare.run_records(run, {})

    def test_three_grnn_records_use_each_methods_sigma_log_and_reference_count(self):
        records = compare.run_records(self.remote_fixture(), {})
        grnn = [record for record in records if record["epochs"] is None]
        self.assertEqual([(record["method"], record["reference_samples"], record["sigma"])
                          for record in grnn], [("RBD", 7, .1), ("ELM", 8, .2), ("SCM", 9, .3)])
        self.assertEqual([record["model"] for record in grnn],
                         ["rbd_grnn_range", "elm_grnn_range", "scm_grnn_range"])
        self.assertTrue(all(record["model_plan_contract"] == "explicit_current" for record in records))

    def test_missing_grnn_or_method_specific_log_does_not_use_legacy_fallback(self):
        run = self.remote_fixture()
        (run / "logs/grnn_RBD_build.log").rename(run / "logs/grnn_build.log")
        with self.assertRaisesRegex(ValueError, "grnn_RBD_build.log"):
            compare.run_records(run, {})
        report = json.loads((run / "report.json").read_text())
        report["metrics"] = [row for row in report["metrics"] if row["model"] not in {"rbd_grnn_range", "elm_grnn_range"}]
        (run / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "canonical suite plan"):
            compare.run_records(run, {})

    def test_sigma_selection_uses_training_cv_mape(self):
        search = [{"sigma": .4, "mean_mae_km": .001, "mean_rmse_km": .002, "mean_mape_pct": .2},
                  {"sigma": .5, "mean_mae_km": .003, "mean_rmse_km": .004, "mean_mape_pct": .1}]
        self.assertEqual(compare.selected_cv_row(search, .5), search[1])
        with self.assertRaisesRegex(ValueError, "mean MAPE optimum"):
            compare.selected_cv_row(search, .4)

    def local_fixture(self, method, *, historical=False):
        run = self.root / f"local-{method}"
        run.mkdir()
        prediction = run / "predictions.csv"
        metrics = self.csv_fixture(prediction)
        search = run / "sigma_search.json"
        search.write_text(json.dumps([{"sigma": .1, "mean_mape_pct": .1,
                                      "mean_mae_km": .001, "mean_rmse_km": .002}]))
        metadata = run / "metadata.json"
        metadata.write_text(json.dumps({"split": {"strategy": "range_overlap", "train_side": "before"}}))
        report = {"schema_version": 1, "run_id": run.name, "status": "PASS",
                  "purpose": "complete_range_overlap_grnn_without_accuracy_acceptance_threshold",
                  "predictions": {"path": str(prediction), **metrics},
                  "sigma_search_path": str(search), "reference": {"sigma": .1, "reference_sample_count": 7},
                  "dataset": {"metadata_file": str(metadata)}, "raw_inputs": [],
                  "source_files": [{"path": str(compare.PROJECT / compare.TASKS_SOURCE),
                                    "sha256": compare.LEGACY_TASKS_SHA256}],
                  "evaluation_scope": {"test_role": "final_evaluation_only"},
                  "config": {"dataset": {"split_strategy": "range_overlap", "train_side": "before"},
                             "validation": {"seed": 2026}, "training": {"epochs": 100},
                             "grnn": {"cv_folds": 5}}}
        if not historical:
            selection = resolve_selection(method, "GRNN")
            report.update(method=method, model=selection["model"], selection=selection)
        (run / "report.json").write_text(json.dumps(report))
        return run

    def test_local_three_method_reports_and_saved_scm_contract(self):
        for method in ("RBD", "ELM", "SCM"):
            records = compare.run_records(self.local_fixture(method), {})
            self.assertEqual([(record["method"], record["model"], record["epochs"]) for record in records],
                             [(method, method.lower() + "_grnn_range", None)])
        legacy = self.local_fixture("SCM-old", historical=True)
        record = compare.run_records(legacy, {})[0]
        self.assertEqual((record["method"], record["model_plan_contract"]), ("SCM", "historical_scm_only"))

    def test_local_mismatched_or_missing_new_method_identity_is_rejected(self):
        run = self.local_fixture("RBD")
        report = json.loads((run / "report.json").read_text())
        report["model"] = "scm_grnn_range"
        (run / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "method/model"):
            compare.run_records(run, {})
        for key in ("method", "model", "selection"):
            report.pop(key)
        report["source_files"][0]["sha256"] = "0"*64
        (run / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "historical source contract"):
            compare.run_records(run, {})

    def merged_records(self, records):
        for record in records:
            (self.root / record["run_id"]).mkdir(exist_ok=True)
            (self.root / record["run_id"] / "report.json").write_text("{}")
        by_run = {r["run_id"]: r for r in records}
        with mock.patch.object(compare, "run_records", side_effect=lambda run, sources: [by_run[run.name]]):
            return compare.merge(self.root)[0]

    def grnn_record(self, run):
        return {"run_id": run, "method": "SCM", "model": "scm_grnn_range",
                "analysis_split": "range_overlap", "train_side": "before", "csv": {"sha256": "same-csv"},
                "raw_input_identity_sha256": "same-inputs", "grnn_computation_config": {"seed": 2026},
                "configured_neural_epochs": 100, "validation_evidence": None, "source_sha256": run}

    def test_same_inputs_and_output_are_kept_as_recomputation_evidence(self):
        old, final = self.grnn_record("old"), self.grnn_record("final")
        final["validation_evidence"] = {"status": "PASS"}
        result = self.merged_records([old, final])
        self.assertEqual([r["run_id"] for r in result["records"]], ["final"])
        duplicate = result["duplicates"][0]
        self.assertEqual(duplicate["duplicate"]["source_sha256"], "old")
        self.assertEqual(duplicate["canonical_run_id"], "final")
        self.assertFalse(duplicate["independent_repeat"])
        self.assertIn("recomputation_consistency", duplicate["reason"])

    def test_different_raw_inputs_or_computation_configs_are_not_folded(self):
        original = self.grnn_record("original")
        changed_input = copy.deepcopy(original)
        changed_input.update(run_id="other-input", raw_input_identity_sha256="different-inputs")
        changed_config = copy.deepcopy(original)
        changed_config.update(run_id="other-config", grnn_computation_config={"seed": 2027})
        result = self.merged_records([original, changed_input, changed_config])
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(result["duplicates"], [])

    def test_feature_methods_do_not_collapse_identical_grnn_csvs(self):
        records = []
        for method in ("RBD", "ELM", "SCM"):
            record = self.grnn_record(method.lower())
            record.update(method=method, model=method.lower() + "_grnn_range")
            records.append(record)
        result = self.merged_records(records)
        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(result["duplicates"], [])


if __name__ == "__main__":
    unittest.main()
