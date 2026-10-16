"""Workflow contracts for scaling experiments; numerical execution stays on haiqin."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.data_scale_pack.localization_networks.workflow import config, tasks, worker

WORKFLOW = Path(__file__).resolve().parents[1]


class ConfigurationAndPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = config.load_config(WORKFLOW / "haiqin_full_interleaved.toml")

    def test_all_presets_have_complete_scale_contract(self):
        for path in sorted(WORKFLOW.glob("*.toml")):
            with self.subTest(path=path.name):
                value = config.load_config(path)
                self.assertEqual(value["scaling"], {"reference_samples": 600, "exponent": 0.5})
                self.assertEqual(value["training"]["train_fraction"], 1.0)
                self.assertNotIn("grnn", value)

    def test_scaling_fields_and_types_are_strict(self):
        cases = [
            ("scaling", "reference_samples", 0), ("scaling", "reference_samples", 1.5),
            ("scaling", "reference_samples", True), ("scaling", "exponent", float("nan")),
            ("scaling", "exponent", 0), ("scaling", "exponent", True),
            ("training", "train_fraction", 0), ("training", "train_fraction", 1.01),
            ("training", "train_fraction", float("inf")), ("training", "train_fraction", True),
            ("validation", "device", "cpu"),
        ]
        for table, key, value in cases:
            with self.subTest(table=table, key=key, value=value):
                bad = copy.deepcopy(self.config)
                bad[table][key] = value
                with self.assertRaises(ValueError):
                    config.validate_config(bad)
        for mutate in (
            lambda value: value.update(grnn={}),
            lambda value: value["scaling"].update(unknown=1),
            lambda value: value["scaling"].pop("exponent"),
        ):
            bad = copy.deepcopy(self.config)
            mutate(bad)
            with self.assertRaises(ValueError):
                config.validate_config(bad)

    def test_plans_are_scale_only_with_bounded_smoke(self):
        full = tasks.model_plan("experiment")
        smoke = tasks.model_plan("smoke")
        self.assertEqual(len(full), 18)
        self.assertEqual(len({(row["method"], row["model"]) for row in full}), 18)
        self.assertEqual(len(smoke), 6)
        self.assertEqual({row["architecture"] for row in smoke}, {"complex_CNN", "complex_CNN_baseline"})
        self.assertEqual(tasks.model_plan("quick"), [])
        for row in full:
            self.assertEqual(tasks.validate_selection(row), row)
            self.assertIn("scripts/data_scale_pack/", tasks.entry_script(row))
        self.assertEqual(len(tasks.model_plan("selected", {"method": "SCM", "network": None})), 6)
        self.assertEqual(len(tasks.model_plan("selected", {"method": None, "network": "complex_CNN_large"})), 3)

    def test_adaptive_frequencies_cannot_use_excluded_training_rows(self):
        for modes in (["adapt"], ["deep", "adapt"]):
            with self.subTest(modes=modes):
                bad = copy.deepcopy(self.config)
                bad["dataset"]["frequency_selection_modes"] = modes
                with self.assertRaisesRegex(ValueError, "retained optimizer-training subset"):
                    config.validate_config(bad)

    def test_removed_families_and_model_crossovers_fail(self):
        for name in ("GRNN", "real_CNN", "ResNet18", "ResNet50"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                tasks.resolve_selection("RBD", name)
        row = tasks.resolve_selection("ELM", "complex_CNN_small")
        row["model"] = "complex_cnn_small_range"
        with self.assertRaises(ValueError):
            tasks.validate_selection(row)
        with self.assertRaises(ValueError):
            tasks.model_plan("smoke", {"method": "RBD", "network": None})


class WorkerScalingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="swellex-scale-worker-")
        self.addCleanup(temporary.cleanup)
        self.instance = worker.Worker(Path(temporary.name) / "run")
        self.instance.config = config.load_config(WORKFLOW / "haiqin_full_interleaved.toml")
        self.instance.config["training"].update(epochs=1, train_fraction=0.5, base_channels=4)
        self.instance.config["scaling"] = {"reference_samples": 100, "exponent": 0.75}
        self.instance.python = sys.executable
        self.instance.dataset_dirs = {"RBD": "/fixture/data"}
        self.directory = self.instance.project / "outputs/networks_results/RBD/complex_cnn_range/train_outputs"
        self.directory.mkdir(parents=True)
        self.checkpoint_path = self.directory / "fixture_best_1015_010203.pt"
        self.checkpoint_path.touch()
        self.scaling = {
            "train_count": 2, "scaling_policy": self.instance.config["scaling"],
            "scaling_policy_version": "train-count-v1", "calibration_status": "uncalibrated",
        }
        self.checkpoint = {
            "pack": worker.PACK, "method": "RBD", "model_name": "complex_cnn_range",
            "train_fraction": 0.5, "scaling": self.scaling,
            "sampling": {"train_fraction": 0.5},
            "train_indices": [0, 2], "val_indices": [1], "epoch": 1,
            "model_config": {"fixture": True},
            "training_inventory": {"fixture": True},
        }
        self.sidecar = {
            "pack": worker.PACK, "method": "RBD", "model": "complex_cnn_range",
            "scaling": self.scaling, "model_config": self.checkpoint["model_config"],
            "sampling": {"train_fraction": 0.5},
            "training_inventory": {"fixture": True},
        }
        self.sidecar_path = self.directory / "fixture_architecture_1015_010203.json"
        self.write_sidecar()

    def write_sidecar(self):
        self.sidecar_path.write_text(json.dumps(self.sidecar))

    def resolve(self):
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(load=lambda *a, **kw: self.checkpoint)}):
            return self.instance.resolved_architecture("RBD", "complex_cnn_range", self.checkpoint_path)

    def test_architecture_is_bound_to_checkpoint_and_training_population(self):
        result = self.resolve()
        self.assertEqual(result["scaling"], self.scaling)
        self.assertEqual(result["architecture_path"], self.sidecar_path.relative_to(self.instance.run).as_posix())
        self.checkpoint["train_indices"] = [0]
        with self.assertRaisesRegex(ValueError, "training count"):
            self.resolve()

    def test_sidecar_or_checkpoint_identity_mismatch_fails(self):
        self.sidecar["pack"] = "base_pack"
        self.write_sidecar()
        with self.assertRaisesRegex(ValueError, "sidecar pack"):
            self.resolve()
        self.sidecar["pack"] = worker.PACK
        self.sidecar["model_config"] = {}
        self.write_sidecar()
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            self.resolve()
        self.sidecar["model_config"] = self.checkpoint["model_config"]
        self.write_sidecar()
        self.checkpoint["method"] = "SCM"
        with self.assertRaisesRegex(ValueError, "Checkpoint pack/method/model"):
            self.resolve()

    def test_training_receives_scaling_controls_and_reports_resolution(self):
        history = [{"epoch": 1, "train_loss": 1., "val_loss": 1., "train_mae_km": 1.,
                    "val_mae_km": 1., "val_rmse_km": 1.}]
        (self.directory / "fixture_history_1015_010203.json").write_text(json.dumps(history))
        predictions = self.directory.parent / "test_outputs"
        predictions.mkdir()
        (predictions / "fixture_predictions_1015_010203.csv").touch()
        with mock.patch.object(self.instance, "phase", side_effect=lambda name, fn: fn()), \
                mock.patch.object(self.instance, "command") as command, \
                mock.patch.object(self.instance, "test_rows", return_value=[]), \
                mock.patch.object(worker, "prediction_metrics", return_value={"mae_km": 1.}), \
                mock.patch.object(self.instance, "resolved_architecture", return_value={"scaling": self.scaling}), \
                mock.patch.object(self.instance, "training_evidence", return_value={"optimizer_steps": 1}) as evidence:
            result = self.instance.train_predict("RBD", "complex_cnn_range")
        arguments = command.call_args_list[0].args[1]
        for key, value in (("--train-fraction", "0.5"), ("--scale-reference-samples", "100"), ("--scale-exponent", "0.75"), ("--base-channels", "4")):
            self.assertEqual(arguments[arguments.index(key) + 1], value)
        self.assertEqual(result["scaling"], self.scaling)
        self.assertEqual(result["train_fraction"], 0.5)
        self.assertEqual(result["optimizer_steps"], 1)
        evidence.assert_called_once()

    def test_resume_receives_same_controls_and_keeps_architecture(self):
        last = self.directory / "fixture_last_1015_010203.pt"
        last.touch()
        destination = self.instance.project / "outputs/resume_validation/RBD"
        destination.mkdir(parents=True)
        resumed_path = destination / "fixture_last_1015_010204.pt"
        resumed_path.touch()
        resumed = {**self.checkpoint, "epoch": 2}

        def load(path, **kwargs):
            return resumed if path == resumed_path else self.checkpoint

        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(load=load)}), \
                mock.patch.object(self.instance, "command") as command:
            self.instance.checkpoint_resume()
            arguments = command.call_args.args[1]
            for key, value in (("--train-fraction", "0.5"), ("--scale-reference-samples", "100"), ("--scale-exponent", "0.75"), ("--base-channels", "4")):
                self.assertEqual(arguments[arguments.index(key) + 1], value)
            resumed["model_config"] = {"fixture": "changed"}
            with self.assertRaisesRegex(ValueError, "frozen architecture"):
                self.instance.checkpoint_resume()


class TrainingHistoryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"epoch": index, "train_loss": 0.2, "val_loss": 0.3, "train_mae_km": 0.1,
             "train_rmse_km": 0.2, "val_mae_km": rmse / 2, "val_rmse_km": rmse,
             "epoch_elapsed_s": 1.5, "optimizer_steps_epoch": 2, "optimizer_steps": index * 2}
            for index, rmse in ((1, 0.4), (2, 0.6))
        ]
        self.best = {"epoch": 1, "metrics": self.history[0], "history": self.history[:1], "best_rmse_km": 0.4}
        self.last = {"epoch": 2, "metrics": self.history[-1], "history": self.history,
                     "best_rmse_km": 0.4, "train_indices": list(range(5)), "scaling": {"parameter_count": 100}}

    def metrics(self):
        return worker.training_history_metrics(self.history, self.best, self.last, train_wall_s=4.0, batch_size=4)

    def test_best_metrics_and_compute_cost_come_from_verified_history(self):
        result = self.metrics()
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["best_val_rmse_km"], 0.4)
        self.assertEqual(result["best_val_mae_km"], 0.2)
        self.assertEqual(result["best_train_rmse_km"], 0.2)
        self.assertEqual(result["sum_epoch_elapsed_s"], 3.0)
        self.assertEqual(result["total_train_wall_s"], 4.0)
        self.assertEqual(result["optimizer_steps"], 4)

    def test_disagreeing_artifact_and_step_evidence_are_rejected(self):
        for key, value in (("epoch", 2), ("metrics", {}), ("history", []), ("best_rmse_km", 0.6)):
            original = self.best[key]
            self.best[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.metrics()
            self.best[key] = original
        self.history[1]["optimizer_steps"] = 3
        with self.assertRaisesRegex(ValueError, "cumulative"):
            self.metrics()
        self.history[1]["optimizer_steps"] = 4
        self.history[1]["optimizer_steps_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "batch geometry"):
            self.metrics()
        self.history[1]["optimizer_steps_epoch"] = 2
        self.history[1]["epoch_elapsed_s"] = 5
        with self.assertRaisesRegex(ValueError, "enclosing training phase"):
            self.metrics()


if __name__ == "__main__":
    unittest.main()
