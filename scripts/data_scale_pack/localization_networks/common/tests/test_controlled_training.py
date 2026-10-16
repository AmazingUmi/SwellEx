"""Remote-only contracts for independent full batches and exact update budgets."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.data_scale_pack.localization_networks.common.checkpoints import validate_checkpoint
from scripts.data_scale_pack.localization_networks.common.cli import build_parser
from scripts.data_scale_pack.localization_networks.common.training_utils import (
    ControlledBatchStream, IndexedTrainingDataset, controlled_budget_plan,
    controlled_sampler_plan, run_epoch,
)
from scripts.data_scale_pack.localization_networks.rbd.network.bindings import BINDINGS
from scripts.data_scale_pack.localization_networks.validation.fixtures import write_fixture


def synthetic_file(path: Path, count: int, offset: int):
    write_fixture(path, "RBD", False)
    rng = np.random.default_rng(100 + offset)
    with h5py.File(path, "r+") as h5:
        attrs = dict(h5["X"].attrs)
        del h5["X"]
        h5["X"] = rng.standard_normal((count, 4, 11, 2)).astype(np.float32)
        h5["X"].attrs.update(attrs)
        for key, values in (
            ("y_range_km", np.linspace(1, 3, count, dtype=np.float32)),
            ("valid_sample", np.ones(count, dtype=np.uint8)),
            ("split/source_segment_idx", np.arange(count) + offset + 1),
            ("time/window_center_s", np.arange(count, dtype=np.float64) + offset + 0.5),
        ):
            del h5[key]
            h5[key] = values


class ControlledSamplingTests(unittest.TestCase):
    def test_cli_preserves_legacy_defaults_and_budget_exclusivity(self):
        parser = build_parser(BINDINGS)
        base = ["train", "--data", "fixture", "--model", "complex_cnn_small_range"]
        args = parser.parse_args(base)
        self.assertEqual(args.epochs, 100)
        self.assertIsNone(args.optimizer_updates)
        self.assertFalse(args.controlled_batches)
        updates = parser.parse_args(base + ["--optimizer-updates", "4200", "--validation-interval-updates", "42", "--controlled-batches"])
        self.assertIsNone(updates.epochs)
        self.assertEqual(updates.optimizer_updates, 4200)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base + ["--epochs", "100", "--optimizer-updates", "4200"])

    def test_full_data_epoch_and_update_plans_are_identical_in_execution(self):
        epochs = controlled_budget_plan(epochs=100, optimizer_updates=None, validation_interval_updates=None, train_count=2700, batch_size=64)
        updates = controlled_budget_plan(epochs=None, optimizer_updates=4200, validation_interval_updates=42, train_count=2700, batch_size=64)
        for key in set(epochs) - {"mode", "requested_epochs", "requested_optimizer_updates"}:
            self.assertEqual(epochs[key], updates[key])
        self.assertEqual(epochs["total_optimizer_updates"], 4200)
        self.assertEqual(epochs["validation_checks"], 100)
        for kwargs in (
            {"epochs": None, "optimizer_updates": 4201, "validation_interval_updates": 42},
            {"epochs": 100, "optimizer_updates": 4200, "validation_interval_updates": 42},
            {"epochs": None, "optimizer_updates": 1, "validation_interval_updates": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                controlled_budget_plan(train_count=2700, batch_size=64, **kwargs)
        with self.assertRaisesRegex(ValueError, "complete training batch"):
            controlled_budget_plan(epochs=2, optimizer_updates=None, validation_interval_updates=None, train_count=7, batch_size=8)

    @staticmethod
    def stream():
        dataset = TensorDataset(torch.arange(20, dtype=torch.float32).view(-1, 1), torch.zeros(20, 1))
        indexed = IndexedTrainingDataset(dataset, list(range(101, 121)))
        seed = controlled_sampler_plan(2026)["train_seed"]
        return ControlledBatchStream(DataLoader(indexed, batch_size=4, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed)))

    def test_sampler_is_independent_of_model_rng_and_record_boundaries(self):
        first = self.stream()
        list(first.take_updates(8))
        torch.manual_seed(44)
        torch.nn.Linear(37, 29)
        torch.rand(1000)
        second = self.stream()
        for _ in range(4):
            list(second.take_updates(2))
            torch.rand(37)
        left, right = first.progress(), second.progress()
        for key in set(left) - {"record_batch_sequence_sha256"}:
            self.assertEqual(left[key], right[key])
        self.assertEqual(left["optimizer_steps"], 8)
        self.assertEqual(left["data_passes_completed"], 1)
        self.assertEqual(left["updates_in_current_pass"], 3)

    def test_validation_loader_and_eval_do_not_advance_model_rng(self):
        dataset = TensorDataset(torch.ones(8, 1), torch.ones(8, 1))
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.5))
        seed = controlled_sampler_plan(2026)["validation_seed"]
        loader = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(seed))
        before = torch.random.get_rng_state().clone()
        run_epoch(model, loader, torch.nn.SmoothL1Loss(), torch.device("cpu"), 0., 1.)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_data_pass_prefixes_include_short_arm_end_outside_validation_grid(self):
        short, long = self.stream(), self.stream()
        list(short.take_updates(5))
        first_boundary = short.progress()["batch_sequence_sha256"]
        list(short.take_updates(5))
        short_progress = short.progress()
        for _ in range(4):
            list(long.take_updates(3))
        long_progress = long.progress()
        self.assertEqual(short_progress["data_pass_batch_sequence_sha256"],
                         [first_boundary, short_progress["batch_sequence_sha256"]])
        self.assertEqual(short_progress["data_pass_batch_sequence_sha256"],
                         long_progress["data_pass_batch_sequence_sha256"][:2])
        self.assertEqual(long_progress["updates_in_current_pass"], 2)
        self.assertNotEqual(long_progress["data_pass_batch_sequence_sha256"][-1], long_progress["batch_sequence_sha256"])


class ControlledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="swellex-controlled-")
        cls.root = Path(cls.temp.name)
        cls.parser = build_parser(BINDINGS)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for count in (24, 32):
            directory = cls.root / f"data{count}"
            directory.mkdir()
            synthetic_file(directory / "fixture_train.h5", count, 0)
            synthetic_file(directory / "fixture_val.h5", 8, 100)
            synthetic_file(directory / "fixture_test.h5", 8, 200)
        try:
            cls.epoch_path, cls.epoch_checkpoint = cls.run_training("epochs", 32, epochs=2)
            cls.update_path, cls.update_checkpoint = cls.run_training("updates", 32, updates=8, interval=4)
            cls.cross_path, cls.cross_checkpoint = cls.run_training("cross-pass", 24, updates=8, interval=2)
        except BaseException:
            torch.set_num_threads(cls.previous_threads)
            cls.temp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.temp.cleanup()

    @classmethod
    def args(cls, name, count, *, epochs=None, updates=None, interval=None, resume=None, controlled=True):
        tokens = ["train", "--data", str(cls.root / f"data{count}"), "--model", "complex_cnn_small_range",
                  "--output-dir", str(cls.root / name), "--batch-size", "8", "--base-channels", "2",
                  "--num-workers", "0", "--device", "cpu", "--seed", "2026"]
        if controlled:
            tokens.append("--controlled-batches")
        if updates is not None:
            tokens.extend(["--optimizer-updates", str(updates), "--validation-interval-updates", str(interval)])
        else:
            tokens.extend(["--epochs", str(epochs)])
        tokens.extend(["--no-resume"] if resume is None else ["--resume-checkpoint", str(resume)])
        return cls.parser.parse_args(tokens)

    @classmethod
    def run_training(cls, name, count, **kwargs):
        args = cls.args(name, count, **kwargs)
        with redirect_stdout(io.StringIO()):
            args.func(args)
        paths = list((cls.root / name).rglob("*_last_*.pt"))
        if len(paths) != 1:
            raise AssertionError("Expected one controlled last checkpoint")
        return paths[0], torch.load(paths[0], map_location="cpu", weights_only=True)

    def test_equal_budget_modes_have_identical_batches_lr_weights_and_validation(self):
        left, right = self.epoch_checkpoint, self.update_checkpoint
        for key in ("initial_model_state_sha256", "sampler_plan", "scheduler_plan", "training_progress"):
            self.assertEqual(left[key], right[key])
        for lrow, rrow in zip(left["history"], right["history"], strict=True):
            self.assertEqual({key: value for key, value in lrow.items() if key != "epoch_elapsed_s"},
                             {key: value for key, value in rrow.items() if key != "epoch_elapsed_s"})
        for key in left["model_state"]:
            torch.testing.assert_close(left["model_state"][key], right["model_state"][key], rtol=0, atol=0)
        self.assertEqual(left["training_progress"]["sample_presentations"], 64)
        self.assertEqual(left["training_progress"]["unique_train_sources_seen"], 32)
        for checkpoint in (left, right):
            validate_checkpoint(checkpoint, method="RBD", model_name="complex_cnn_small_range", dataset_name=str(self.root / "data32"))

    def test_exact_updates_cross_data_pass_boundaries_and_stop_without_extra_batch(self):
        checkpoint = self.cross_checkpoint
        history = checkpoint["history"]
        self.assertEqual([row["optimizer_steps"] for row in history], [2, 4, 6, 8])
        self.assertEqual([row["data_passes_completed"] for row in history], [0, 1, 2, 2])
        self.assertEqual([row["updates_in_current_pass"] for row in history], [2, 1, 0, 2])
        self.assertEqual([row["sample_presentations_record"] for row in history], [16] * 4)
        self.assertEqual(checkpoint["training_progress"]["sample_presentations"], 64)
        self.assertEqual(checkpoint["training_progress"]["unique_train_sources_seen"], 24)
        self.assertEqual(len(checkpoint["training_progress"]["data_pass_batch_sequence_sha256"]), 2)
        self.assertTrue(all("data_pass_batch_sequence_sha256" not in row for row in history))
        self.assertEqual(history[-1]["next_lr"], 0)
        self.assertTrue(all(row["record_unit"] == "validation_checkpoint" for row in history))

    def test_resume_is_rejected_for_controlled_request_and_controlled_checkpoint(self):
        for controlled in (True, False):
            args = self.args("rejected", 32, epochs=1, resume=self.epoch_path, controlled=controlled)
            with self.subTest(controlled=controlled), self.assertRaisesRegex(ValueError, "sampler/RNG state"):
                args.func(args)

    def test_controlled_evidence_tampering_is_rejected_and_sidecar_has_complete_recipe(self):
        checkpoint = self.epoch_checkpoint
        for section, key, value in (
            ("budget_plan", "total_optimizer_updates", 9), ("sampler_plan", "train_seed", 2026),
            ("scheduler_plan", "step_unit", "epoch"), ("training_progress", "sample_presentations", 63),
        ):
            broken = deepcopy(checkpoint)
            broken[section][key] = value
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_checkpoint(broken, method="RBD", model_name="complex_cnn_small_range", dataset_name=str(self.root / "data32"))
        sidecars = list((self.root / "epochs").rglob("*_architecture_*.json"))
        self.assertEqual(len(sidecars), 1)
        saved = json.loads(sidecars[0].read_text())
        for key in ("budget_plan", "sampler_plan", "scheduler_plan", "initial_model_state_sha256"):
            self.assertEqual(saved[key], checkpoint[key])

    def test_controlled_learning_rate_history_and_state_are_verified(self):
        for section, key, value in (
            ("history", "last_update_lr", 0.75), ("history", "lr_sequence_sha256", "0" * 64),
            ("scheduler_state", "last_epoch", 7), ("optimizer_state", "lr", 0.75),
        ):
            broken = deepcopy(self.epoch_checkpoint)
            if section == "history":
                broken[section][0][key] = value
            elif section == "optimizer_state":
                broken[section]["param_groups"][0][key] = value
            else:
                broken[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_checkpoint(broken, method="RBD", model_name="complex_cnn_small_range", dataset_name=str(self.root / "data32"))

    def test_data_pass_prefix_evidence_rejects_missing_invalid_and_inconsistent_digests(self):
        checkpoint = self.epoch_checkpoint
        progress = checkpoint["training_progress"]
        self.assertEqual(len(progress["data_pass_batch_sequence_sha256"]), progress["data_passes_completed"])
        self.assertEqual(progress["data_pass_batch_sequence_sha256"][-1], progress["batch_sequence_sha256"])
        for value in ([], ["invalid", progress["batch_sequence_sha256"]], ["0"*64, "0"*64]):
            broken = deepcopy(checkpoint)
            broken["training_progress"]["data_pass_batch_sequence_sha256"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "data-pass batch"):
                validate_checkpoint(broken, method="RBD", model_name="complex_cnn_small_range", dataset_name=str(self.root / "data32"))

    def test_controlled_prediction_reconstructs_saved_layout(self):
        args = self.parser.parse_args(["predict", "--data", str(self.root / "data32"), "--model", "complex_cnn_small_range",
                                       "--checkpoint", self.epoch_checkpoint["best_checkpoint_path"],
                                       "--output-dir", str(self.root / "prediction"), "--device", "cpu", "--no-plot", "--print-first", "0"])
        with redirect_stdout(io.StringIO()):
            args.func(args)
        self.assertEqual(len(list((self.root / "prediction").rglob("*_predictions_*.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
