"""Standard-library contracts for opt-in controlled training budgets."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from scripts.data_scale_pack.localization_networks.workflow import config, remote, tasks, worker


class BudgetConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = config.load_config(remote.DEFAULT_CONFIG)

    def test_legacy_configuration_is_not_mutated_or_extended(self):
        before = json.dumps(self.config, sort_keys=True)
        self.assertEqual(config.validate_config(self.config), self.config)
        self.assertEqual(json.dumps(self.config, sort_keys=True), before)
        self.assertFalse(set(self.config["training"]) & config.TRAINING_EXTENSIONS)
        legacy = worker.Worker(Path("/tmp/budget-unit-fixture"))
        legacy.config = self.config
        self.assertEqual(legacy.budget_arguments(), ["--epochs", str(self.config["training"]["epochs"])])

    def test_controlled_epoch_and_update_arguments_are_unambiguous(self):
        instance = worker.Worker(Path("/tmp/budget-unit-fixture"))
        instance.config = self.config
        self.config["training"].update(epochs=100, controlled_batches=True,
                                       optimizer_updates=None, validation_interval_updates=None)
        config.validate_config(self.config)
        self.assertEqual(instance.budget_arguments(), ["--epochs", "100", "--controlled-batches"])
        self.config["training"].update(epochs=None, optimizer_updates=4200, validation_interval_updates=42)
        config.validate_config(self.config)
        self.assertEqual(instance.budget_arguments(), ["--optimizer-updates", "4200", "--validation-interval-updates", "42", "--controlled-batches"])

    def test_invalid_training_combinations_are_rejected(self):
        cases = [dict(epochs=None), dict(epochs=0), dict(optimizer_updates=10),
                 dict(validation_interval_updates=2), dict(controlled_batches="true"),
                 dict(epochs=None, optimizer_updates=10, validation_interval_updates=2, controlled_batches=False),
                 dict(epochs=2, optimizer_updates=10, validation_interval_updates=2, controlled_batches=True),
                 dict(epochs=None, optimizer_updates=10, validation_interval_updates=3, controlled_batches=True),
                 dict(epochs=None, optimizer_updates=True, validation_interval_updates=1, controlled_batches=True),
                 dict(optimizer_steps=10)]
        for values in cases:
            value = copy.deepcopy(self.config)
            value["training"].update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                config.validate_config(value)

    def test_calibration_reference_is_exact_and_relative(self):
        valid = {"path": "scripts/data_scale_pack/localization_networks/workflow/calibration.json", "sha256": "a" * 64}
        self.config["calibration"] = valid
        config.validate_config(self.config)
        for value in ({**valid, "extra": True}, {**valid, "sha256": "A" * 64},
                      {**valid, "path": "/tmp/calibration.json"},
                      {**valid, "path": "scripts/data_scale_pack/localization_networks/workflow/../calibration.json"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.validate_calibration(value)

    def test_budget_plan_contains_five_fixed_models_and_no_legacy_resume(self):
        filters = {"method": "SCM", "network": None}
        jobs = tasks.model_plan("budget", filters)
        self.assertEqual(len(jobs), 5)
        self.assertEqual([row["architecture"] for row in jobs], list(tasks.BUDGET_ARCHITECTURES))
        self.assertNotIn("complex_CNN", tasks.BUDGET_ARCHITECTURES)
        phases = tasks.expected_phases("budget", filters)
        self.assertNotIn("checkpoint_resume", phases)
        self.assertNotIn("python_quick", phases)
        self.assertIn("visualization", phases)
        for filters in ({"method": None, "network": None}, {"method": "SCM", "network": "complex_CNN"}):
            with self.assertRaises(ValueError):
                tasks.model_plan("budget", filters)

    def test_full_fraction_100_epochs_matches_4200_updates_with_drop_last(self):
        training = {**self.config["training"], "batch_size": 64, "epochs": 100,
                    "controlled_batches": True, "optimizer_updates": None, "validation_interval_updates": None}
        epoch = worker.expected_controlled_budget(training, 2700)
        update = worker.expected_controlled_budget({**training, "epochs": None, "optimizer_updates": 4200,
                                                    "validation_interval_updates": 42}, 2700)
        for key in ("total_optimizer_updates", "validation_interval_updates", "validation_checks", "steps_per_data_pass", "drop_last"):
            self.assertEqual(epoch[key], update[key])
        self.assertEqual(epoch["total_optimizer_updates"], 4200)
        with self.assertRaisesRegex(ValueError, "complete batch"):
            worker.expected_controlled_budget(training, 32)

    def test_controlled_smoke_cannot_reach_unsupported_legacy_resume(self):
        self.config["training"]["controlled_batches"] = True
        with mock.patch.object(remote.subprocess, "run") as ssh, self.assertRaisesRegex(ValueError, "legacy resume"):
            remote.dispatch(self.config, "smoke", False)
        ssh.assert_not_called()

    def test_model_initialization_cannot_change_the_controlled_batch_stream(self):
        row = {key: "fixture" for key in ("budget_plan", "sampler_plan", "scheduler_plan", "train_identity_sha256",
               "val_identity_sha256", "test_identity_sha256", "final_batch_sequence_sha256", "final_lr_sequence_sha256",
               "final_seen_source_identity_sha256", "unique_train_sources_seen", "sample_presentations")}
        row["scaling"] = {"train_count": 96}
        row["data_pass_batch_sequence_sha256"] = ["a" * 64, "b" * 64]
        rows = [copy.deepcopy(row) for _ in range(5)]
        worker.validate_controlled_comparison(rows)
        for key in ("final_batch_sequence_sha256", "train_identity_sha256", "final_lr_sequence_sha256"):
            rows[-1][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Controlled models differ"):
                worker.validate_controlled_comparison(rows)
            rows[-1][key] = "fixture"
        rows[-1]["data_pass_batch_sequence_sha256"][0] = "c" * 64
        with self.assertRaisesRegex(ValueError, "Controlled models differ"):
            worker.validate_controlled_comparison(rows)


class BudgetHistoryTests(unittest.TestCase):
    def setUp(self):
        self.training = {"epochs": None, "optimizer_updates": 6, "validation_interval_updates": 3,
                         "controlled_batches": True, "batch_size": 4}
        self.budget = worker.expected_controlled_budget(self.training, 8)
        self.history = []
        for index, rmse in ((1, .4), (2, .6)):
            updates = index * 3
            self.history.append({"epoch": index, "train_loss": .2, "val_loss": .3, "train_mae_km": .1,
                                 "train_rmse_km": .2, "val_mae_km": rmse / 2, "val_rmse_km": rmse,
                                 "epoch_elapsed_s": 1.5, "optimizer_steps_epoch": 3, "optimizer_steps": updates,
                                 "record_unit": "validation_checkpoint", "validation_index": index,
                                 "sample_presentations_record": 12, "sample_presentations": updates * 4,
                                 "data_passes_completed": updates // 2, "updates_in_current_pass": updates % 2,
                                 "unique_train_sources_seen": 8, "batch_sequence_sha256": "a" * 64,
                                 "record_batch_sequence_sha256": "b" * 64, "lr_sequence_sha256": "c" * 64,
                                 "seen_source_identity_sha256": "e" * 64})
        metadata = {"budget_plan": self.budget, "sampler_plan": {"fixture": True},
                    "scheduler_plan": {"fixture": True}, "initial_model_state_sha256": "d" * 64}
        self.best = {**metadata, "epoch": 1, "metrics": self.history[0], "history": self.history[:1], "best_rmse_km": .4}
        self.last = {**metadata, "epoch": 2, "metrics": self.history[-1], "history": self.history,
                     "best_rmse_km": .4, "train_indices": list(range(8)), "scaling": {"parameter_count": 100},
                     "training_progress": {"data_pass_batch_sequence_sha256": ["1" * 64, "2" * 64, "a" * 64]}}

    def metrics(self):
        return worker.training_history_metrics(self.history, self.best, self.last, train_wall_s=4., batch_size=4,
                                               training_config=self.training)

    def test_history_records_updates_not_data_passes(self):
        result = self.metrics()
        self.assertEqual(result["optimizer_steps"], 6)
        self.assertEqual(result["best_validation_index"], 1)
        self.assertEqual(result["best_optimizer_step"], 3)
        self.assertEqual(result["validation_checks"], 2)
        self.assertEqual(result["sample_presentations"], 24)
        self.assertEqual(result["history_record_unit"], "validation_checkpoint")
        self.assertEqual(result["final_seen_source_identity_sha256"], "e" * 64)
        self.assertEqual(result["data_pass_batch_sequence_sha256"], ["1" * 64, "2" * 64, "a" * 64])
        self.assertIsNot(result["data_pass_batch_sequence_sha256"], self.last["training_progress"]["data_pass_batch_sequence_sha256"])

    def test_data_pass_prefixes_require_every_completed_pass_and_final_stream(self):
        for hashes in (None, ["a" * 64], ["1" * 64, "invalid", "a" * 64], ["1" * 64, "2" * 64, "b" * 64]):
            self.last["training_progress"]["data_pass_batch_sequence_sha256"] = hashes
            with self.subTest(hashes=hashes), self.assertRaisesRegex(ValueError, "data-pass batch-prefix"):
                self.metrics()

    def test_incomplete_coverage_bad_counters_and_digest_are_rejected(self):
        for key, value in (("unique_train_sources_seen", 7), ("sample_presentations", 23),
                           ("data_passes_completed", 2), ("validation_index", 4),
                           ("batch_sequence_sha256", "not-a-hash")):
            previous = self.history[-1][key]
            self.history[-1][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.metrics()
            self.history[-1][key] = previous

    def test_frozen_budget_and_metadata_cannot_be_swapped(self):
        self.best["budget_plan"] = {**self.budget, "total_optimizer_updates": 9}
        with self.assertRaisesRegex(ValueError, "frozen training"):
            self.metrics()
        self.best["budget_plan"] = self.budget
        self.best["sampler_plan"] = {"fixture": False}
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            self.metrics()
        with self.assertRaisesRegex(ValueError, "legacy epoch"):
            worker.training_history_metrics(self.history, self.best, self.last, train_wall_s=4., batch_size=4)


if __name__ == "__main__":
    unittest.main()
