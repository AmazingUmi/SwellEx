"""Strict evidence identity and scale-aware plots for the new experiment pack."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.data_scale_pack.localization_networks.workflow.config import load_config
from scripts.data_scale_pack.localization_networks.workflow.remote import DEFAULT_CONFIG
from scripts.data_scale_pack.localization_networks.workflow.tasks import model_plan
from scripts.data_scale_pack.visualization.results import ModelResult, load_run, read_predictions, summarize
from scripts.data_scale_pack.visualization.plot_results import prediction_grid_layout, render_run
from scripts.run_identity import make_run_id


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ScaleResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swellex-scale-plot-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = load_config(DEFAULT_CONFIG)
        self.config["training"].update(train_fraction=0.5, epochs=1)
        self.selection = {"method": "RBD", "network": "complex_CNN"}
        self.run = self.root / make_run_id("data_scale_pack", self.config, self.selection)
        self.run.mkdir()
        self.job = model_plan("selected", self.selection)[0]
        self.directory = self.run / "project/outputs/networks_results/RBD" / self.job["model"]
        self.csv = ("sample,source_segment_idx,window_center_s,pred_range_km,true_range_km,abs_error_km\n"
                    "0,1,1.0,2.1,2.0,0.1\n1,2,2.0,2.8,3.0,0.2\n").encode()
        prediction = self.directory / "test_outputs/model_predictions_1015_120000.csv"
        prediction.parent.mkdir(parents=True)
        prediction.write_bytes(self.csv)
        self.scaling = {
            "train_count": 128, "scaling_policy_version": "train-count-v1",
            "scaling_policy": copy.deepcopy(self.config["scaling"]),
            "selection_mode": "auto", "selected_profile": "small",
            "stage_repeats": [1, 1, 1], "stage_strides": [[1, 1], [1, 2], [1, 2]],
            "parameter_count": 12000, "parameter_budget": 14000.0,
            "calibration_status": "uncalibrated",
        }
        self.architecture = {
            "pack": "data_scale_pack", "method": "RBD", "model": self.job["model"],
            "scaling": copy.deepcopy(self.scaling),
            "model_config": {**{key: copy.deepcopy(value) for key, value in self.scaling.items() if key != "calibration_status"},
                             "input_elements": 21, "input_freq_bins": 22, "dropout": 0.0,
                             "base_channels": self.config["training"]["base_channels"]},
            "sampling": {"version": "temporal-block-prefix-v1", "train_fraction": 0.5,
                         "candidate_train_count": 256, "train_count": 128, "validation_count": 64,
                         "block_rows": 32, "seed": self.config["validation"]["seed"], "source_identity_sha256": "a" * 64},
        }
        self.architecture_path = self.directory / "train_outputs/model_architecture_1015_120000.json"
        write_json(self.architecture_path, self.architecture)
        history = [{"epoch": 1, "loss_space": "km", "lr": 0.001, "train_loss": 0.1, "val_loss": 0.1,
                    "train_mae_km": 0.1, "val_mae_km": 0.1, "train_rmse_km": 0.15, "val_rmse_km": 0.15}]
        write_json(self.directory / "train_outputs/model_history_1015_120000.json", history)
        inventory = [{"path": "scripts/data_scale_pack/localization_networks/workflow/tasks.py", "sha256": "1" * 64, "size": 1}]
        source_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.manifest = {"schema_version": 1, "pack": "data_scale_pack", "run_id": self.run.name,
                         "suite": "selected", "selection": self.selection, "files": inventory,
                         "source_sha256": source_hash, "config_sha256": config_hash}
        self.request = {"pack": "data_scale_pack", "run_id": self.run.name, "suite": "selected",
                        "selection": self.selection, "config": self.config}
        metrics = {**summarize(read_predictions(self.csv)), "method": "RBD", "model": self.job["model"],
                   "epochs": 1, "scaling": self.scaling, "train_fraction": 0.5,
                   "architecture_path": self.architecture_path.relative_to(self.run).as_posix()}
        self.report = {"schema_version": 1, "pack": "data_scale_pack", "run_id": self.run.name,
                       "suite": "selected", "status": "PASS", "selection": self.selection,
                       "purpose": "configured_model_selection_without_accuracy_acceptance_threshold",
                       "source_sha256": source_hash, "config_sha256": config_hash,
                       "model_plan": [self.job], "metrics": [metrics],
                       "checks": [{"name": f"evaluate_RBD_{self.job['model']}", "status": "PASS"}]}
        self.save()

    def save(self):
        for filename, value in (("report.json", self.report), ("request.json", self.request), ("manifest.json", self.manifest)):
            write_json(self.run / filename, value)

    def test_reader_captures_scaling_config_and_hashes_architecture_input(self):
        data = load_run(self.run)
        self.assertEqual(data.models[0].architecture, self.architecture)
        self.assertEqual(data.models[0].label, "RBD/auto small")
        self.assertIn("Ntrain=128", data.models[0].caption)
        self.assertEqual(data.config, self.config)
        paths = {item["path"] for item in data.sources}
        self.assertIn(self.architecture_path.relative_to(self.run).as_posix(), paths)
        self.assertTrue({"request.json", "manifest.json", "report.json"}.issubset(paths))

    def test_missing_or_foreign_pack_is_rejected(self):
        for pack in (None, "base_pack", "joint_feature_pack"):
            with self.subTest(pack=pack):
                if pack is None:
                    self.report.pop("pack", None)
                else:
                    self.report["pack"] = pack
                self.save()
                with self.assertRaisesRegex(ValueError, "another experiment pack"):
                    load_run(self.run)

    def test_missing_or_modified_plan_cannot_fall_back_to_historical_contract(self):
        for plan in (None, [], [{**self.job, "architecture": "GRNN"}]):
            with self.subTest(plan=plan):
                if plan is None:
                    self.report.pop("model_plan", None)
                else:
                    self.report["model_plan"] = plan
                self.save()
                with self.assertRaisesRegex(ValueError, "model plan"):
                    load_run(self.run)

    def test_configuration_source_and_selection_tampering_are_rejected(self):
        original = copy.deepcopy((self.request, self.manifest))
        mutations = (
            lambda: self.request["config"]["validation"].update(seed=999),
            lambda: self.manifest["files"][0].update(sha256="2" * 64),
            lambda: self.manifest.update(selection={"method": "RBD", "network": "complex_CNN_small"}),
            lambda: self.request.update(pack="base_pack"),
        )
        for mutation in mutations:
            self.request, self.manifest = copy.deepcopy(original)
            mutation()
            self.save()
            with self.assertRaises(ValueError):
                load_run(self.run)

    def test_architecture_identity_or_report_disagreement_is_rejected(self):
        mutations = (
            lambda value: value.update(pack="base_pack"),
            lambda value: value.update(model="complex_cnn_large_range"),
            lambda value: value["scaling"].update(train_count=129),
            lambda value: value["sampling"].update(train_fraction=1.0),
        )
        for mutation in mutations:
            sidecar = copy.deepcopy(self.architecture)
            mutation(sidecar)
            write_json(self.architecture_path, sidecar)
            with self.subTest(sidecar=sidecar), self.assertRaises(ValueError):
                load_run(self.run)

    def test_filtering_does_not_allow_partial_or_damaged_recorded_plan(self):
        self.report["suite"] = self.request["suite"] = self.manifest["suite"] = "experiment"
        self.report["model_plan"] = model_plan("experiment")
        self.report["purpose"] = "configured_experiment_without_accuracy_acceptance_threshold"
        for record in (self.report, self.request, self.manifest):
            record.pop("selection", None)
        self.save()
        with self.assertRaises(ValueError):
            load_run(self.run, models=["RBD/complex_cnn_range"])

    def test_canonical_complete_grid_has_three_methods_and_six_profiles(self):
        models = [ModelResult(job["method"], job["model"], [], [], {}, {}) for job in model_plan("experiment")]
        grid = prediction_grid_layout(models)
        self.assertEqual(len(grid), 3)
        self.assertTrue(all(len(row) == 6 for row in grid))
        self.assertEqual(sorted(index for row in grid for index in row), list(range(18)))

    def test_render_exports_frozen_scale_and_configuration_identity(self):
        output = self.root / "figures"
        manifest = render_run(self.run, output, dpi=72)
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["pack"], row["selected_profile"], row["train_count"], row["parameter_count"]),
                         ("data_scale_pack", "small", "128", "12000"))
        self.assertEqual(row["train_fraction"], "0.5")
        self.assertEqual(manifest["config_sha256"], self.report["config_sha256"])
        self.assertEqual(manifest["models"][0]["scaling"], self.scaling)
        self.assertEqual(manifest["model_plan"], [self.job])
        self.assertNotIn("grnn", manifest["transformations"])
        for item in manifest["files"]:
            payload = (output / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
