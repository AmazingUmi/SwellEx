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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.joint_feature_pack.visualization.results import (ModelResult,
                                          PREDICTION_FIELDS, load_run, read_predictions, summarize)
from scripts.joint_feature_pack.visualization.plot_results import main, prediction_grid_layout, render_run
from scripts.joint_feature_pack.localization_networks.workflow.tasks import METHODS, model_plan, resolve_selection


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
        self.assertEqual(prediction_grid_layout(self.models(model_plan("experiment"))),
                         [[4*i, 4*i+1, 4*i+2, 4*i+3, 28+i] for i in range(7)])

    def test_smoke_plan_has_a_grnn_for_each_feature(self):
        self.assertEqual(prediction_grid_layout(self.models(model_plan("smoke"))),
                         [[i, 7+i] for i in range(7)])

    def test_selected_subset_aligns_architectures_without_reordering_models(self):
        by_name = {model.name: model for model in self.models(model_plan("experiment"))}
        names = ["RBD_ELM_SCM/rbd_elm_scm_grnn_range", "RBD/joint_resnet50_range",
                 "RBD_ELM_SCM/joint_complex_cnn_range", "ELM/joint_complex_cnn_range"]
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
        self.report = {"schema_version": 1, "pack": "joint_feature_pack", "fusion": {"mode": "block_rms"},
                       "run_id": self.run.name, "status": "PASS",
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
        selection = resolve_selection("RBD_ELM_SCM", "GRNN" if grnn else "complex_CNN")
        self.report.update(suite="selected",
                           purpose="configured_model_selection_without_accuracy_acceptance_threshold",
                           selection={"method": selection["method"], "network": selection["architecture"]})
        self.report["metrics"] = [row for row in self.report["metrics"]
                                  if row["method"] == selection["method"] and row["model"] == selection["model"]]
        self.report["model_plan"] = model_plan("selected", self.report["selection"])
        self.save_report()

    def test_selected_neural_run_acceptance_and_render(self):
        self.select_report()
        data = load_run(self.run)
        self.assertEqual([model.name for model in data.models], ["RBD_ELM_SCM/joint_complex_cnn_range"])
        self.assertEqual(len(data.models[0].history), 1)
        self.assertEqual(data.models[0].sigma_search, [])
        output = self.root/"selected-neural"
        result = render_run(self.run, output, dpi=72)
        self.assertEqual(len(result["figures"]), 7)
        self.assertTrue((output/"models/RBD_ELM_SCM/joint_complex_cnn_range/training/history.pdf").is_file())
        self.assertFalse((output/"grnn").exists())
        manifest = json.loads((output/"manifest.json").read_text())
        self.assertEqual(manifest["suite"], "selected")
        self.assertEqual(manifest["pack"], "joint_feature_pack")
        self.assertEqual(manifest["fusion"], {"mode": "block_rms"})
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual((output/"source-report.json").read_bytes(), (self.run/"report.json").read_bytes())

    def test_selected_grnn_run_acceptance_and_render(self):
        self.select_report(grnn=True)
        data = load_run(self.run)
        self.assertEqual([model.name for model in data.models], ["RBD_ELM_SCM/rbd_elm_scm_grnn_range"])
        self.assertEqual(data.models[0].history, [])
        self.assertEqual(len(data.models[0].sigma_search), 2)
        output = self.root/"selected-grnn"
        result = render_run(self.run, output, dpi=72)
        self.assertEqual(len(result["figures"]), 7)
        self.assertTrue((output/"grnn/RBD-ELM-SCM-rbd-elm-scm-grnn-range-sigma.pdf").is_file())
        self.assertFalse((output/"training").exists())
        manifest = json.loads((output/"manifest.json").read_text())
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual(manifest["purpose"], self.report["purpose"])

    def test_network_filter_accepts_all_methods_and_rejects_partial_or_duplicate_metrics(self):
        selection = {"method": None, "network": "GRNN"}
        plan = model_plan("selected", selection)
        self.report.update(suite="selected", selection=selection, model_plan=plan,
                           purpose="configured_model_selection_without_accuracy_acceptance_threshold")
        self.populate_plan(plan)
        self.save_report()
        self.assertEqual(len(load_run(self.run).models), len(plan))
        metrics = list(self.report["metrics"])
        for invalid in (metrics[:-1], metrics[:-1] + [metrics[0]]):
            self.report["metrics"] = invalid
            self.save_report()
            with self.assertRaises(ValueError):
                load_run(self.run)

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
        for models in (None, ["RBD/joint_complex_cnn_range"]):
            with self.subTest(models=models), self.assertRaisesRegex(ValueError, "complete filtered plan"):
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



    def test_complete_joint_suite_plans(self):
        for suite, expected in (("smoke", 14), ("experiment", 35)):
            self.report.update(suite=suite, model_plan=model_plan(suite))
            self.populate_plan(model_plan(suite))
            self.save_report()
            data = load_run(self.run)
            self.assertEqual(len(data.models), expected)
            self.assertEqual(sum(bool(model.sigma_search) for model in data.models), 7)


    def test_missing_or_tampered_plan_is_rejected(self):
        original = self.report["model_plan"]
        for plan in (None, original[:-1], [*original, original[0]]):
            self.report["model_plan"] = plan
            self.save_report()
            with self.assertRaisesRegex(ValueError, "joint pack contract"):
                load_run(self.run)

    def test_base_missing_pack_and_unknown_fusion_are_rejected(self):
        for pack in (None, "base_pack", "unknown"):
            self.report["pack"] = pack
            self.save_report()
            with self.subTest(pack=pack), self.assertRaisesRegex(ValueError, "pack mismatch"):
                load_run(self.run)
        self.report["pack"] = "joint_feature_pack"
        for fusion in (None, {}, {"mode": "unknown"}, {"mode": []},
                       {"mode": "block_rms", "extra": True}):
            self.report["fusion"] = fusion
            self.save_report()
            with self.subTest(fusion=fusion), self.assertRaisesRegex(ValueError, "fusion identity"):
                load_run(self.run)
        self.report["fusion"] = {"mode": "raw_concat"}
        self.save_report()
        self.assertEqual(load_run(self.run).report["fusion"], {"mode": "raw_concat"})

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
        self.assertEqual(len(load_run(self.run, ["RBD_ELM_SCM/rbd_elm_scm_grnn_range"]).models), 1)
        for models in (["other/model"], [], ["RBD/joint_complex_cnn_range"]*2):
            with self.assertRaises(ValueError): load_run(self.run, models)

    def test_incomplete_or_unknown_suite_models_rejected_before_selection(self):
        original = list(self.report["metrics"])
        for metrics in (original[:-1], [{**original[0], "model": "unregistered"}, *original[1:]]):
            self.report["metrics"] = metrics
            self.save_report()
            with self.assertRaisesRegex(ValueError, "canonical suite plan"):
                load_run(self.run, ["SCM/joint_complex_cnn_range"])

    def test_unselected_inputs_and_evaluation_are_verified(self):
        selected = ["RBD_ELM_SCM/rbd_elm_scm_grnn_range"]
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
        ecdf_panels = []
        original_savefig = Figure.savefig

        def inspect_savefig(fig, filename, *args, **kwargs):
            stem = Path(filename).stem
            if stem in {"range-time", "prediction-scatter", "residuals-time"}:
                comparison_panels[stem] = [
                    (ax.get_title(), ax.get_visible(), ax.get_subplotspec().rowspan.start,
                     ax.get_subplotspec().colspan.start) for ax in fig.axes
                ]
            if stem == "absolute-error-ecdf":
                self.assertIn("joint_feature_pack | block_rms", fig._suptitle.get_text())
                ecdf_panels.extend((ax.get_title(), [line.get_label() for line in ax.lines])
                                   for ax in fig.axes)
            return original_savefig(fig, filename, *args, **kwargs)

        with patch.object(Figure, "savefig", inspect_savefig):
            result = render_run(self.run, output, dpi=100)
        self.assertEqual(set(comparison_panels), {"range-time", "prediction-scatter", "residuals-time"})
        self.assertEqual(ecdf_panels, [("complex_CNN", list(METHODS)), ("GRNN", list(METHODS))])
        for stem, panels in comparison_panels.items():
            with self.subTest(figure=stem):
                expected = [(f"{method}/{resolve_selection(method, architecture)['model']}", row, column)
                            for row, method in enumerate(METHODS)
                            for column, architecture in enumerate(("complex_CNN", "GRNN"))]
                self.assertEqual([(name, row, column) for name, visible, row, column in panels if visible], expected)
                self.assertEqual([(row, column) for _, visible, row, column in panels if not visible],
                                 [])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["figures"]), 33)
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
        self.assertTrue((output/"metrics.csv").read_text().startswith("pack,fusion,method,model,samples,"))
        with self.assertRaisesRegex(ValueError, "already exists"): render_run(self.run, output)

    def test_ecdf_shared_limits_include_larger_errors_in_later_architecture(self):
        from matplotlib.figure import Figure

        plan = model_plan("experiment")
        self.report.update(suite="experiment", model_plan=plan)
        self.populate_plan(plan)
        later = resolve_selection("RBD_ELM_SCM", "ResNet50")
        later_index = plan.index(later)
        larger_rows = [[0, 5, 4.5, 12.0, 2.0, 10.0], [1, 10, 9.5, 23.0, 3.0, 20.0]]
        self.csv_paths[later_index].write_bytes(csv_bytes(larger_rows))
        self.report["metrics"][later_index].update(summarize(read_predictions(csv_bytes(larger_rows))))
        self.save_report()
        inspected = []
        original_savefig = Figure.savefig

        def inspect_savefig(fig, filename, *args, **kwargs):
            if Path(filename).stem == "absolute-error-ecdf":
                fig.canvas.draw()
                limits = [ax.get_xlim() for ax in fig.axes]
                self.assertEqual(limits[0], limits[1])
                self.assertEqual(limits[0][0], 0)
                self.assertGreater(limits[0][1], 20)
                for ax in fig.axes:
                    self.assertEqual(len(ax.lines), 1)
                    line = ax.lines[0]
                    low, high = ax.get_xlim()
                    self.assertTrue(all(low <= value < high for value in line.get_xdata()))
                    self.assertEqual(line.get_ydata()[-1], 1)
                    end_x, end_y = ax.transData.transform((line.get_xdata()[-1], 1))
                    radius = line.get_markersize() * fig.dpi / 144
                    self.assertGreater(ax.bbox.xmax - end_x, radius)
                    self.assertGreater(ax.bbox.ymax - end_y, radius)
                    inspected.append(ax.get_title())
            return original_savefig(fig, filename, *args, **kwargs)

        selected = ["RBD/joint_complex_cnn_range", "RBD_ELM_SCM/joint_resnet50_range"]
        with patch.object(Figure, "savefig", inspect_savefig):
            render_run(self.run, self.root/"wide-ecdf-figure", dpi=72, models=selected)
        self.assertEqual(inspected, ["complex_CNN", "ResNet50"])

    def test_nearly_flat_sigma_search_keeps_markers_clear_of_the_top_boundary(self):
        from matplotlib.figure import Figure

        self.select_report(grnn=True)
        search = next((self.run/"project/outputs/networks_results/RBD_ELM_SCM/rbd_elm_scm_grnn_range").rglob("*sigma_search*.json"))
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
        self.assertEqual(main(["--run", str(self.run), "--output-dir", str(output), "--model", "RBD_ELM_SCM/rbd_elm_scm_grnn_range", "--dpi", "72"]), 0)
        result = json.loads((output/"manifest.json").read_text())
        self.assertEqual(len(result["models"]), 1)
        self.assertFalse((output/"training").exists())


if __name__ == "__main__":
    unittest.main()
