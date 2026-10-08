"""Result-integrity and export checks; execute on haiqin via the quick suite."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.visualization.results import PREDICTION_FIELDS, load_run, read_predictions, summarize
from scripts.visualization.plot_results import main, render_run
from scripts.localization_networks.workflow.tasks import model_plan


def csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(PREDICTION_FIELDS)
    writer.writerows(rows)
    return handle.getvalue().encode()


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swellex-visualization-")
        self.root = Path(self.temp.name)
        self.run = self.root/"20261008T120000Z-12345678"
        self.run.mkdir()
        self.rows = [[0, 5, 4.5, 2.2, 2.0, 0.2], [1, 10, 9.5, 2.5, 3.0, 0.5]]
        self.history = [{"epoch": 1, "lr": 0.01, "loss_space": "km", "train_loss": 0.2,
                         "val_loss": 0.3, "train_mae_km": 0.2, "val_mae_km": 0.3,
                         "train_rmse_km": 0.3, "val_rmse_km": 0.4}]
        self.report = {"schema_version": 1, "run_id": self.run.name, "status": "PASS",
                       "suite": "smoke", "purpose": "engineering_validation", "checks": [], "metrics": []}
        self.csv_paths = []
        for item in model_plan("smoke"):
            method, model = item["method"], item["model"]
            directory = self.run/"project/outputs/networks_results"/method/model
            prediction = directory/"test_outputs/example_predictions_1008_120000.csv"
            prediction.parent.mkdir(parents=True)
            prediction.write_bytes(csv_bytes(self.rows))
            self.csv_paths.append(prediction)
            metric = {"method": method, "model": model, **summarize(read_predictions(csv_bytes(self.rows)))}
            if item["architecture"] != "GRNN":
                metric["epochs"] = 1
                history_path = directory/"train_outputs/example_history_1008_120000.json"
                history_path.parent.mkdir()
                history_path.write_text(json.dumps(self.history))
                if method == "RBD":
                    self.history_path = history_path
                phase = f"evaluate_{method}_{model}"
            else:
                search = directory/"reference_outputs/example_sigma_search_1008_120000.json"
                search.parent.mkdir()
                search.write_text(json.dumps([
                    {"sigma": 0.1, "mean_mae_km": 0.2, "mean_rmse_km": 0.3, "mean_mape_pct": 2.0},
                    {"sigma": 1.0, "mean_mae_km": 0.4, "mean_rmse_km": 0.5, "mean_mape_pct": 4.0},
                ]))
                phase = "evaluate_grnn"
            self.report["metrics"].append(metric)
            self.report["checks"].append({"name": phase, "status": "PASS"})
        self.save_report()

    def tearDown(self):
        self.temp.cleanup()

    def save_report(self):
        (self.run/"report.json").write_text(json.dumps(self.report))

    def select_report(self, *, grnn=False):
        self.report.update(
            suite="selected",
            purpose="configured_single_model_without_accuracy_acceptance_threshold",
            selection={"method": "SCM" if grnn else "RBD",
                       "architecture": "GRNN" if grnn else "complex_CNN",
                       "model": "scm_grnn_range" if grnn else "complex_cnn_range"},
        )
        self.report["metrics"] = [self.report["metrics"][-1 if grnn else 0]]
        self.save_report()

    def test_selected_neural_run_acceptance_and_render(self):
        self.select_report()
        data = load_run(self.run)
        self.assertEqual([model.name for model in data.models], ["RBD/complex_cnn_range"])
        self.assertEqual(len(data.models[0].history), 1)
        self.assertEqual(data.models[0].sigma_search, [])
        output = self.root/"selected-neural"
        result = render_run(self.run, output, dpi=72)
        self.assertEqual(len(result["figures"]), 7)
        self.assertTrue((output/"models/RBD/complex_cnn_range/training/history.pdf").is_file())
        self.assertFalse((output/"grnn").exists())
        manifest = json.loads((output/"manifest.json").read_text())
        self.assertEqual(manifest["suite"], "selected")
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual((output/"source-report.json").read_bytes(), (self.run/"report.json").read_bytes())

    def test_selected_grnn_run_acceptance_and_render(self):
        self.select_report(grnn=True)
        data = load_run(self.run)
        self.assertEqual([model.name for model in data.models], ["SCM/scm_grnn_range"])
        self.assertEqual(data.models[0].history, [])
        self.assertEqual(len(data.models[0].sigma_search), 2)
        output = self.root/"selected-grnn"
        result = render_run(self.run, output, dpi=72)
        self.assertEqual(len(result["figures"]), 7)
        self.assertTrue((output/"grnn/SCM-scm-grnn-range-sigma.pdf").is_file())
        self.assertFalse((output/"training").exists())
        manifest = json.loads((output/"manifest.json").read_text())
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual(manifest["purpose"], self.report["purpose"])

    def test_selected_missing_or_tampered_selection_rejected(self):
        self.select_report()
        canonical = dict(self.report["selection"])
        invalid = [None, {"method": "RBD", "architecture": "complex_CNN"},
                   {**canonical, "model": "real_cnn_range"},
                   {**canonical, "architecture": "complex_cnn"},
                   {**canonical, "extra": "unexpected"},
                   {"method": "RBD", "architecture": "GRNN", "model": "scm_grnn_range"},
                   {"method": "ELM", "architecture": "complex_CNN", "model": "elm_complex_cnn_range"}]
        for selection in invalid:
            with self.subTest(selection=selection):
                self.report["selection"] = selection
                self.save_report()
                with self.assertRaises(ValueError): load_run(self.run)
        del self.report["selection"]
        self.save_report()
        with self.assertRaises(ValueError): load_run(self.run)

    def test_selected_extra_metrics_rejected_before_model_filter(self):
        extra = dict(self.report["metrics"][1])
        self.select_report()
        self.report["metrics"].append(extra)
        self.save_report()
        for models in (None, ["RBD/complex_cnn_range"]):
            with self.subTest(models=models), self.assertRaisesRegex(ValueError, "exactly one metric"):
                load_run(self.run, models)

    def test_selected_tampered_purpose_rejected(self):
        self.select_report()
        self.report["purpose"] = "engineering_validation"
        self.save_report()
        with self.assertRaisesRegex(ValueError, "purpose"): load_run(self.run)

    def test_canonical_run_and_resume_history_exclusion(self):
        unrelated = self.run/"project/outputs/resume_validation/bad_history_1008_130000.json"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("invalid")
        data = load_run(self.run)
        self.assertEqual([m.name for m in data.models], [f"{row['method']}/{row['model']}" for row in model_plan("smoke")])
        self.assertEqual(data.models[0].metrics["samples"], 2)
        self.assertAlmostEqual(data.models[0].metrics["mae_km"], 0.35)
        self.assertAlmostEqual(data.models[0].metrics["bias_km"], -0.15)
        self.assertFalse(any("resume_validation" in x["path"] for x in data.sources))

    def test_failed_and_quick_reports_rejected(self):
        for key, value in (("status", "FAIL"), ("suite", "quick")):
            original = self.report[key]
            self.report[key] = value
            self.save_report()
            with self.assertRaises(ValueError): load_run(self.run)
            self.report[key] = original

    def test_worker_running_requires_completed_model_phase(self):
        self.report["status"] = "RUNNING"
        self.save_report()
        with self.assertRaises(ValueError): load_run(self.run)
        load_run(self.run, require_pass=False)
        self.report["checks"][0]["status"] = "RUNNING"
        self.save_report()
        with self.assertRaises(ValueError): load_run(self.run, require_pass=False)

    def test_metrics_mismatch_rejected(self):
        self.report["metrics"][0]["mae_km"] += 0.01
        self.save_report()
        with self.assertRaisesRegex(ValueError, "mae_km"): load_run(self.run)

    def test_boolean_schema_and_missing_evaluation_rejected(self):
        self.report["schema_version"] = True
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Unsupported"): load_run(self.run)
        self.report["schema_version"] = 1
        self.report["checks"] = []
        self.save_report()
        with self.assertRaisesRegex(ValueError, "evaluation"): load_run(self.run)

    def test_same_indices_but_shifted_times_rejected(self):
        rows = [list(row) for row in self.rows]
        rows[0][2] += 1
        self.csv_paths[1].write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "identical test"): load_run(self.run)

    def test_same_errors_but_shifted_labels_rejected(self):
        rows = [list(row) for row in self.rows]
        rows[0][3] += 1
        rows[0][4] += 1
        self.csv_paths[1].write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "identical test"): load_run(self.run)

    def test_duplicate_csv_is_not_silently_selected(self):
        shutil.copyfile(self.csv_paths[0], self.csv_paths[0].with_name("duplicate_predictions_1008_130000.csv"))
        with self.assertRaisesRegex(ValueError, "Expected one"): load_run(self.run)

    def test_nonfinite_errors_indices_and_times_rejected(self):
        for column, value in ((3, "nan"), (5, -1), (1, 5), (2, 4.5), (0, 3)):
            rows = [list(row) for row in self.rows]
            rows[1][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                read_predictions(csv_bytes(rows))
        rows = [list(row) for row in self.rows]
        rows[0][5] = 0.1
        with self.assertRaisesRegex(ValueError, "absolute error"): read_predictions(csv_bytes(rows))

    def test_six_decimal_rounding_tolerance(self):
        rows = [[0, 1, 0.5, 1.123457, 1.0, 0.123456]]
        self.assertEqual(len(read_predictions(csv_bytes(rows))), 1)

    def test_loss_space_and_nonfinite_history_rejected(self):
        for key, value in (("loss_space", "unknown"), ("train_rmse_km", float("nan")), ("epoch", True)):
            history = [dict(self.history[0])]
            history[0][key] = value
            self.history_path.write_text(json.dumps(history))
            with self.subTest(key=key), self.assertRaises(ValueError): load_run(self.run)

    def test_mixed_loss_space_and_missing_epochs_rejected(self):
        history = [dict(self.history[0]), {**self.history[0], "epoch": 2, "loss_space": "normalized"}]
        self.report["metrics"][0]["epochs"] = 2
        self.save_report()
        self.history_path.write_text(json.dumps(history))
        with self.assertRaisesRegex(ValueError, "loss space"): load_run(self.run)
        history[1].update(epoch=3, loss_space="km")
        self.history_path.write_text(json.dumps(history))
        with self.assertRaisesRegex(ValueError, "epochs"): load_run(self.run)

    def test_grnn_search_cannot_mix_cv_and_validation(self):
        path = next(self.run.rglob("*sigma_search*.json"))
        rows = json.loads(path.read_text())
        rows[1] = {key.removeprefix("mean_"): value for key, value in rows[1].items()}
        path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, "mixes CV"): load_run(self.run)

    def test_selection_validation(self):
        self.assertEqual(len(load_run(self.run, ["SCM/scm_grnn_range"]).models), 1)
        for models in (["other/model"], [], ["RBD/complex_cnn_range"]*2):
            with self.assertRaises(ValueError): load_run(self.run, models)

    def test_incomplete_or_unknown_suite_models_rejected_before_selection(self):
        original = list(self.report["metrics"])
        for metrics in (original[:-1], [{**original[0], "model": "unregistered"}, *original[1:]]):
            self.report["metrics"] = metrics
            self.save_report()
            with self.assertRaisesRegex(ValueError, "canonical suite plan"):
                load_run(self.run, ["SCM/scm_complex_cnn_range"])

    def test_unselected_inputs_and_evaluation_are_verified(self):
        selected = ["SCM/scm_grnn_range"]
        original = self.csv_paths[0].read_bytes()
        self.csv_paths[0].write_text("invalid CSV")
        with self.assertRaisesRegex(ValueError, "canonical six columns"):
            load_run(self.run, selected)
        self.csv_paths[0].write_bytes(original)
        self.report["checks"][0]["status"] = "RUNNING"
        self.save_report()
        with self.assertRaisesRegex(ValueError, "evaluation has not passed"):
            load_run(self.run, selected)

    def test_external_input_rejected(self):
        outside = self.root/"outside.csv"
        self.csv_paths[0].replace(outside)
        self.csv_paths[0].symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "external"): load_run(self.run)

    def test_rendered_artifacts_dimensions_and_no_overwrite(self):
        output = self.root/"figure"
        result = render_run(self.run, output, dpi=100)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["figures"]), 13)
        manifest = json.loads((output/"manifest.json").read_text())
        self.assertEqual(manifest["models"][0]["samples"], 2)
        self.assertEqual({item["group"] for item in manifest["files"]}, {"data", "figure"})
        self.assertEqual((output/"source-report.json").read_bytes(), (self.run/"report.json").read_bytes())
        self.assertFalse(list(output.rglob("*.png")))
        self.assertFalse((output/"training").exists())
        self.assertEqual(manifest["formats"], ["pdf"])
        for figure in result["figures"]:
            raw = (output/figure).read_bytes()
            self.assertTrue(raw.startswith(b"%PDF-"))
            self.assertTrue(raw.rstrip().endswith(b"%%EOF"))
        for item in model_plan("smoke"):
            folder = output/"models"/item["method"]/item["model"]
            self.assertTrue((folder/"test_outputs/prediction.pdf").is_file())
            if item["architecture"] != "GRNN":
                self.assertTrue((folder/"training/history.pdf").is_file())
        self.assertEqual(len(manifest["renderer_sources"]), 3)
        self.assertTrue((output/"metrics.csv").read_text().startswith("method,model,samples,"))
        with self.assertRaisesRegex(ValueError, "already exists"): render_run(self.run, output)

    def test_cli_model_selection(self):
        output = self.root/"cli-figure"
        self.assertEqual(main(["--run", str(self.run), "--output-dir", str(output), "--model", "SCM/scm_grnn_range", "--dpi", "72"]), 0)
        result = json.loads((output/"manifest.json").read_text())
        self.assertEqual(len(result["models"]), 1)
        self.assertFalse((output/"training").exists())


if __name__ == "__main__":
    unittest.main()
