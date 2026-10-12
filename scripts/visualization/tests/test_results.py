"""Result-integrity and export checks; execute on haiqin via the quick suite."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.visualization.results import (LEGACY_TASKS_SHA256, TASKS_SOURCE, ModelResult,
                                          PREDICTION_FIELDS, load_run, read_predictions, summarize)
from scripts.visualization.plot_results import main, prediction_grid_layout, render_run
from scripts.localization_networks.workflow.tasks import model_plan


def csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow(PREDICTION_FIELDS)
    writer.writerows(rows)
    return handle.getvalue().encode()


class PredictionGridTests(unittest.TestCase):
    def models(self, plan):
        return [ModelResult(item["method"], item["model"], [], [], [], {}) for item in plan]

    def test_full_plan_keeps_each_feature_in_one_row(self):
        models = self.models(model_plan("experiment"))
        self.assertEqual(prediction_grid_layout(models), [
            [0, 1, 2, 3, 12],
            [4, 5, 6, 7, 13],
            [8, 9, 10, 11, 14],
        ])

    def test_smoke_plan_has_a_grnn_for_each_feature(self):
        self.assertEqual(prediction_grid_layout(self.models(model_plan("smoke"))), [
            [0, 3], [1, 4], [2, 5],
        ])

    def test_selected_subset_aligns_architectures_without_reordering_models(self):
        by_name = {model.name: model for model in self.models(model_plan("experiment"))}
        names = ["SCM/scm_grnn_range", "RBD/resnet50_range",
                 "SCM/scm_complex_cnn_range", "ELM/elm_complex_cnn_range"]
        models = [by_name[name] for name in names]
        self.assertEqual(prediction_grid_layout(models), [
            [None, 1, None], [3, None, None], [2, None, 0],
        ])
        self.assertEqual([model.name for model in models], names)

    def test_single_selected_model_has_one_panel(self):
        for model in self.models(model_plan("experiment")):
            with self.subTest(model=model.name):
                self.assertEqual(prediction_grid_layout([model]), [[0]])


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
                       "suite": "smoke", "purpose": "engineering_validation", "checks": [], "metrics": [],
                       "model_plan": model_plan("smoke")}
        self.populate_plan(model_plan("smoke"))
        self.save_report()

    def populate_plan(self, plan):
        self.csv_paths = []
        self.report["metrics"], self.report["checks"] = [], []
        for item in plan:
            method, model = item["method"], item["model"]
            directory = self.run/"project/outputs/networks_results"/method/model
            prediction = directory/"test_outputs/example_predictions_1008_120000.csv"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_bytes(csv_bytes(self.rows))
            self.csv_paths.append(prediction)
            metric = {"method": method, "model": model, **summarize(read_predictions(csv_bytes(self.rows)))}
            if item["architecture"] != "GRNN":
                metric["epochs"] = 1
                history_path = directory/"train_outputs/example_history_1008_120000.json"
                history_path.parent.mkdir(exist_ok=True)
                history_path.write_text(json.dumps(self.history))
                if method == "RBD":
                    self.history_path = history_path
                phase = f"evaluate_{method}_{model}"
            else:
                search = directory/"reference_outputs/example_sigma_search_1008_120000.json"
                search.parent.mkdir(exist_ok=True)
                search.write_text(json.dumps([
                    {"sigma": 0.1, "mean_mae_km": 0.2, "mean_rmse_km": 0.3, "mean_mape_pct": 2.0},
                    {"sigma": 1.0, "mean_mae_km": 0.4, "mean_rmse_km": 0.5, "mean_mape_pct": 4.0},
                ]))
                phase = f"evaluate_{method}_{model}"
            self.report["metrics"].append(metric)
            self.report["checks"].append({"name": phase, "status": "PASS"})

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
        self.report["model_plan"] = model_plan("selected", self.report["selection"])
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

    def historical_manifest(self, *, tasks_sha256=LEGACY_TASKS_SHA256):
        inventory = [{"path": TASKS_SOURCE, "size": 1, "sha256": tasks_sha256}]
        digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = {"schema_version": 1, "run_id": self.run.name, "suite": self.report["suite"],
                    "files": inventory, "source_sha256": digest}
        if self.report["suite"] == "selected":
            manifest["selection"] = self.report["selection"]
        self.report["source_sha256"] = digest
        (self.run/"manifest.json").write_text(json.dumps(manifest))

    def use_historical_plan(self, suite):
        self.report["suite"] = suite
        planned = [item for item in model_plan(suite, self.report.get("selection"))
                   if item["architecture"] != "GRNN" or item["method"] == "SCM"]
        self.populate_plan(planned)
        self.report.pop("model_plan", None)
        for check in self.report["checks"]:
            if check["name"] == "evaluate_SCM_scm_grnn_range":
                check["name"] = "evaluate_grnn"
        self.historical_manifest()
        self.save_report()
        return planned

    def test_complete_current_and_historical_suite_plans(self):
        for suite, expected in (("smoke", 6), ("experiment", 15)):
            self.report.update(suite=suite, model_plan=model_plan(suite))
            self.populate_plan(model_plan(suite))
            self.save_report()
            self.assertEqual(len(load_run(self.run).models), expected)
            self.assertEqual(sum(bool(model.sigma_search) for model in load_run(self.run).models), 3)
        for suite, expected in (("smoke", 4), ("experiment", 13)):
            self.use_historical_plan(suite)
            self.assertEqual(len(load_run(self.run).models), expected)
            self.report["metrics"].pop()
            self.save_report()
            with self.assertRaisesRegex(ValueError, "canonical suite plan"):
                load_run(self.run)

    def test_historical_selected_scm_grnn_and_source_binding(self):
        self.select_report(grnn=True)
        self.use_historical_plan("selected")
        self.assertEqual(load_run(self.run).models[0].name, "SCM/scm_grnn_range")
        self.historical_manifest(tasks_sha256="0"*64)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "historical source contract"):
            load_run(self.run)

    def test_missing_or_tampered_current_plan_never_falls_back_to_old_count(self):
        current = model_plan("smoke")
        self.report["model_plan"] = [item for item in current if item["method"] == "SCM" or item["architecture"] != "GRNN"]
        self.report["metrics"] = [metric for metric in self.report["metrics"]
                                  if metric["method"] == "SCM" or not metric["model"].endswith("grnn_range")]
        self.save_report()
        with self.assertRaisesRegex(ValueError, "canonical suite plan"):
            load_run(self.run)
        del self.report["model_plan"]
        self.historical_manifest(tasks_sha256="0"*64)
        self.save_report()
        with self.assertRaisesRegex(ValueError, "historical source contract"):
            load_run(self.run)

    def test_current_grnn_requires_method_specific_evaluation(self):
        self.report["checks"] = [check for check in self.report["checks"]
                                 if check["name"] != "evaluate_RBD_rbd_grnn_range"]
        self.report["checks"].append({"name": "evaluate_grnn", "status": "PASS"})
        self.save_report()
        with self.assertRaisesRegex(ValueError, "evaluate_RBD_rbd_grnn_range"):
            load_run(self.run)

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
        from matplotlib.figure import Figure

        output = self.root/"figure"
        comparison_panels = {}
        original_savefig = Figure.savefig

        def inspect_savefig(fig, filename, *args, **kwargs):
            stem = Path(filename).stem
            if stem in {"range-time", "prediction-scatter", "residuals-time"}:
                comparison_panels[stem] = [
                    (ax.get_title(), ax.get_visible(), ax.get_subplotspec().rowspan.start,
                     ax.get_subplotspec().colspan.start) for ax in fig.axes
                ]
            return original_savefig(fig, filename, *args, **kwargs)

        with patch.object(Figure, "savefig", inspect_savefig):
            result = render_run(self.run, output, dpi=100)
        self.assertEqual(set(comparison_panels), {"range-time", "prediction-scatter", "residuals-time"})
        for stem, panels in comparison_panels.items():
            with self.subTest(figure=stem):
                self.assertEqual([(name, row, column) for name, visible, row, column in panels if visible], [
                    ("RBD/complex_cnn_range", 0, 0), ("RBD/rbd_grnn_range", 0, 1),
                    ("ELM/elm_complex_cnn_range", 1, 0), ("ELM/elm_grnn_range", 1, 1),
                    ("SCM/scm_complex_cnn_range", 2, 0), ("SCM/scm_grnn_range", 2, 1),
                ])
                self.assertEqual([(row, column) for _, visible, row, column in panels if not visible],
                                 [])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["figures"]), 17)
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
            else:
                stem = f"{item['method']}-{item['model']}".replace("_", "-")
                self.assertTrue((output/f"grnn/{stem}-sigma.pdf").is_file())
                self.assertFalse((folder/"training/history.pdf").exists())
        self.assertEqual(len(manifest["renderer_sources"]), 5)
        self.assertTrue((output/"metrics.csv").read_text().startswith("method,model,samples,"))
        with self.assertRaisesRegex(ValueError, "already exists"): render_run(self.run, output)

    def test_nearly_flat_sigma_search_keeps_markers_clear_of_the_top_boundary(self):
        from matplotlib.figure import Figure

        self.select_report(grnn=True)
        search = next((self.run/"project/outputs/networks_results/SCM/scm_grnn_range").rglob("*sigma_search*.json"))
        search.write_text(json.dumps([
            {"sigma": sigma, "mean_mae_km": .2+offset, "mean_rmse_km": 0,
             "mean_mape_pct": 2+offset}
            for sigma, offset in ((.1, 0), (1, 1e-10))
        ]))
        inspected = []
        original_savefig = Figure.savefig

        def inspect_savefig(fig, filename, *args, **kwargs):
            if Path(filename).stem.endswith("-sigma"):
                fig.canvas.draw()
                for ax in fig.axes:
                    low, high = ax.get_ylim()
                    data_lines = [line for line in ax.lines if line.get_marker() == "o"]
                    self.assertEqual(len(data_lines), 1)
                    line = data_lines[0]
                    self.assertEqual(low, 0)
                    self.assertTrue(all(value < high for value in line.get_ydata()))
                    # Check visible marker clearance in display units; merely
                    # exceeding the data by roundoff would still clip the dots.
                    top_pixel = max(ax.transData.transform((x, y))[1]
                                    for x, y in zip(line.get_xdata(), line.get_ydata()))
                    marker_radius = line.get_markersize()*fig.dpi/144
                    self.assertGreater(ax.bbox.ymax-top_pixel, marker_radius)
                    inspected.append(ax.get_ylabel())
            return original_savefig(fig, filename, *args, **kwargs)

        with patch.object(Figure, "savefig", inspect_savefig):
            render_run(self.run, self.root/"flat-sigma-figure", dpi=72)
        self.assertEqual(inspected, ["MAE (km)", "RMSE (km)", "MAPE (%)"])

    def test_cli_model_selection(self):
        output = self.root/"cli-figure"
        self.assertEqual(main(["--run", str(self.run), "--output-dir", str(output), "--model", "SCM/scm_grnn_range", "--dpi", "72"]), 0)
        result = json.loads((output/"manifest.json").read_text())
        self.assertEqual(len(result["models"]), 1)
        self.assertFalse((output/"training").exists())


if __name__ == "__main__":
    unittest.main()
