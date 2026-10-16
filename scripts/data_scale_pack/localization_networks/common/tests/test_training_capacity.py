"""Remote-only tiny end-to-end training, restoration, and prediction contracts.

The test process explicitly uses CPU for these fixtures. The enclosing remote
quick suite separately requires CUDA; this is not a runtime fallback.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import torch

from scripts.data_scale_pack.localization_networks.common import scaling
from scripts.data_scale_pack.localization_networks.common.checkpoints import validate_checkpoint
from scripts.data_scale_pack.localization_networks.common.cli import build_parser
from scripts.data_scale_pack.localization_networks.common.data_selection import sample_indices, source_identity
from scripts.data_scale_pack.localization_networks.common.training import range_coverage, split_inventory
from scripts.data_scale_pack.localization_networks.rbd.network.bindings import BINDINGS
from scripts.data_scale_pack.localization_networks.validation.fixtures import write_fixture
from scripts.data_scale_pack.localization_networks.workflow.worker import Worker


class TrainingCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="swellex-capacity-contract-")
        cls.root = Path(cls.temp.name)
        cls.dataset_dir = cls.root / "fixture"
        cls.dataset_dir.mkdir()
        cls.train_path = cls.dataset_dir / "fixture_train.h5"
        cls.test_path = cls.dataset_dir / "fixture_test.h5"
        write_fixture(cls.train_path, "RBD", False)
        write_fixture(cls.test_path, "RBD", False)
        with h5py.File(cls.test_path, "r+") as h5:
            h5["split/source_segment_idx"][...] += 100
            h5["time/window_center_s"][...] += 100
        cls.original_train_bytes = cls.train_path.read_bytes()
        cls.parser = build_parser(BINDINGS)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            cls.base_path, cls.checkpoint = cls.run_training("initial")
        except BaseException:
            torch.set_num_threads(cls.previous_threads)
            cls.temp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        cls.temp.cleanup()

    def tearDown(self):
        # Source identity hashes bytes, so restoring the fixture also restores
        # its identity without relying on modification time or test order.
        self.train_path.write_bytes(self.original_train_bytes)
        (self.dataset_dir / "fixture_val.h5").unlink(missing_ok=True)

    @classmethod
    def train_args(cls, output_name, *, resume=None, fraction=0.5, exponent=2.0):
        tokens = [
            "train", "--data", str(cls.dataset_dir),
            "--model", "complex_cnn_range", "--output-dir", str(cls.root / output_name),
            "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
            "--base-channels", "2", "--dropout", "0", "--val-fraction", "0.25",
            "--train-fraction", str(fraction), "--scale-reference-samples", "2",
            "--scale-exponent", str(exponent), "--seed", "2026",
        ]
        tokens.extend(["--no-resume"] if resume is None else ["--resume-checkpoint", str(resume)])
        return cls.parser.parse_args(tokens)

    @classmethod
    def run_training(cls, output_name, **kwargs):
        args = cls.train_args(output_name, **kwargs)
        with redirect_stdout(io.StringIO()):
            args.func(args)
        checkpoints = list((cls.root / output_name).rglob("*_last_*.pt"))
        if len(checkpoints) != 1:
            raise AssertionError(f"Expected one last checkpoint, found {checkpoints}")
        return checkpoints[0], torch.load(checkpoints[0], map_location="cpu")

    def test_resolved_count_excludes_invalid_validation_and_unselected_rows(self):
        checkpoint = self.checkpoint
        self.assertEqual(checkpoint["pack"], "data_scale_pack")
        self.assertEqual(len(checkpoint["train_indices"]), 2)
        self.assertEqual(len(checkpoint["val_indices"]), 1)
        self.assertEqual(checkpoint["sampling"]["candidate_train_count"], 4)
        self.assertEqual(checkpoint["sampling"]["train_count"], 2)
        self.assertEqual(checkpoint["sampling"]["validation_count"], 1)
        self.assertEqual(checkpoint["model_config"]["train_count"], 2)
        self.assertEqual(checkpoint["scaling"]["selected_profile"], "medium")
        self.assertTrue(set(checkpoint["train_indices"]).isdisjoint(checkpoint["val_indices"]))
        with h5py.File(self.train_path, "r") as h5:
            self.assertEqual(h5["X"].shape[0], 7)
        sidecars = list((self.root / "initial").rglob("*_architecture_*.json"))
        self.assertEqual(len(sidecars), 1)
        sidecar = json.loads(sidecars[0].read_text())
        self.assertEqual(sidecar["model_config"], checkpoint["model_config"])
        self.assertEqual(sidecar["scaling"], checkpoint["scaling"])
        self.assertEqual(sidecar["sampling"], checkpoint["sampling"])
        self.assertEqual(sidecar["training_inventory"], checkpoint["training_inventory"])
        self.assertEqual(checkpoint["training_inventory"]["train"]["count"], 2)
        self.assertEqual(checkpoint["training_inventory"]["validation"]["count"], 1)
        self.assertGreater(checkpoint["metrics"]["epoch_elapsed_s"], 0)
        self.assertEqual(checkpoint["metrics"]["optimizer_steps_epoch"], 1)
        self.assertEqual(checkpoint["metrics"]["optimizer_steps"], 1)

    def test_fraction_changes_training_membership_without_changing_validation(self):
        _, full = self.run_training("full-budget", fraction=1.0)
        partial = self.checkpoint
        self.assertEqual(full["val_indices"], partial["val_indices"])
        self.assertEqual(full["sampling"]["validation_count"], partial["sampling"]["validation_count"])
        self.assertLess(set(partial["train_indices"]), set(full["train_indices"]))
        self.assertEqual(full["model_config"]["train_count"], 4)
        self.assertEqual(full["scaling"]["selected_profile"], "xlarge")
        self.assertEqual(full["training_inventory"]["validation"], partial["training_inventory"]["validation"])
        self.assertLess(set(partial["training_inventory"]["train"]["source_segment_indices"]), set(full["training_inventory"]["train"]["source_segment_indices"]))
        indices, source_ids, times = list(range(10, 110)), list(range(1, 121)), list(range(120))
        selected = [set(sample_indices(indices, source_ids, times, fraction, 2026)) for fraction in (0.25, 0.5, 1.0)]
        self.assertLess(selected[0], selected[1])
        self.assertLess(selected[1], selected[2])
        self.assertEqual([len(subset) for subset in selected], [25, 50, 100])
        self.assertTrue(all(subset.isdisjoint(set(range(10))) for subset in selected))

    def test_missing_or_duplicated_source_identity_is_rejected_before_training(self):
        for key in ("split/source_segment_idx", "time/window_center_s"):
            self.train_path.write_bytes(self.original_train_bytes)
            with h5py.File(self.train_path, "r+") as h5:
                del h5[key]
            args = self.train_args("missing-metadata")
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "requires"):
                args.func(args)
        self.train_path.write_bytes(self.original_train_bytes)
        with h5py.File(self.train_path, "r+") as h5:
            # Raw rows 0 and 2 both survive valid-label filtering.
            h5["split/source_segment_idx"][2] = h5["split/source_segment_idx"][0]
        args = self.train_args("duplicate-source")
        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            args.func(args)
        self.train_path.write_bytes(self.original_train_bytes)
        write_fixture(self.dataset_dir / "fixture_val.h5", "RBD", False)
        args = self.train_args("overlapping-explicit-validation")
        with self.assertRaisesRegex(ValueError, "overlap or repeat"):
            args.func(args)

    def test_source_byte_change_at_same_path_size_and_mtime_rejects_resume(self):
        before = self.train_path.stat()
        original = source_identity([self.train_path], None)
        with h5py.File(self.train_path, "r+") as h5:
            h5["X"][0, 0, 0, 0] += 0.125
        os.utime(self.train_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.train_path.stat().st_size, before.st_size)
        self.assertEqual(self.train_path.stat().st_mtime_ns, before.st_mtime_ns)
        changed = source_identity([self.train_path], None)
        self.assertNotEqual(original["train_sources"][0]["sha256"], changed["train_sources"][0]["sha256"])
        args = self.train_args("changed-bytes", resume=self.base_path)
        with self.assertRaisesRegex(ValueError, "source_identity"):
            args.func(args)

    def test_checkpoint_pack_mismatch_is_rejected(self):
        checkpoint = deepcopy(self.checkpoint)
        checkpoint["pack"] = "base_pack"
        with self.assertRaisesRegex(ValueError, "pack"):
            validate_checkpoint(checkpoint, method="RBD", model_name="complex_cnn_range", dataset_name=str(self.dataset_dir))

    def test_resume_restores_architecture_and_rejects_changed_policy_or_budget(self):
        with patch.object(BINDINGS.model, "build_model_config", side_effect=AssertionError("Resume reselected capacity")):
            _, resumed = self.run_training("resumed", resume=self.base_path)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["model_config"], self.checkpoint["model_config"])
        self.assertEqual(resumed["scaling"], self.checkpoint["scaling"])
        self.assertEqual(resumed["sampling"], self.checkpoint["sampling"])
        self.assertEqual(resumed["training_inventory"], self.checkpoint["training_inventory"])
        self.assertEqual(resumed["metrics"]["optimizer_steps"], 2)
        for changes, message in (({"exponent": 3}, "capacity policy"), ({"fraction": 1}, "sampling")):
            args = self.train_args("invalid-resume", resume=self.base_path, **changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                args.func(args)

    def test_prediction_uses_saved_capacity_with_different_heldout_count(self):
        args = self.parser.parse_args([
            "predict", "--data", str(self.dataset_dir), "--model", "complex_cnn_range",
            "--checkpoint", str(self.base_path), "--output-dir", str(self.root / "prediction"),
            "--batch-size", "8", "--device", "cpu", "--num-workers", "0", "--print-first", "0", "--no-plot",
        ])
        with (
            patch.object(BINDINGS.model, "build_model_config", side_effect=AssertionError("Prediction selected capacity")),
            patch.object(scaling, "resolve_capacity", side_effect=AssertionError("Prediction resolved capacity")),
            redirect_stdout(io.StringIO()),
        ):
            args.func(args)
        predictions = list((self.root / "prediction").rglob("*_predictions_*.csv"))
        self.assertEqual(len(predictions), 1)
        with predictions[0].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(int(row["source_segment_idx"]) >= 151 for row in rows))
        self.assertEqual(self.checkpoint["model_config"]["train_count"], 2)

    def test_split_inventory_hash_order_and_linear_label_quantiles(self):
        rows = [(index, float(index * 10), float(index)) for index in range(1, 5)]
        result = split_inventory(rows)
        self.assertEqual(result, split_inventory(reversed(rows)))
        self.assertEqual(result["range_km_summary"], {"min": 1., "q25": 1.75, "median": 2.5, "q75": 3.25, "max": 4.})
        heldout = split_inventory([(5, 50., 0.), (6, 60., 2.), (7, 70., 5.)])
        self.assertEqual(range_coverage(result, heldout), {"within_train_range_count": 1, "within_train_range_fraction": 1 / 3})
        modified = rows.copy()
        modified[0] = (1, 10., 1.001)
        self.assertNotEqual(result["identity_sha256"], split_inventory(modified)["identity_sha256"])

    def test_worker_exports_verified_full_precision_membership_and_training_cost(self):
        instance = Worker(self.root / "evidence-run")
        instance.config = {"training": {"batch_size": 8}, "validation": {"seed": 2026}}
        instance.dataset_dirs = {"RBD": str(self.dataset_dir)}
        history = self.checkpoint["history"]
        instance.report["checks"] = [{"name": "train_RBD_complex_cnn_range", "status": "PASS",
                                      "elapsed_s": sum(row["epoch_elapsed_s"] for row in history) + 1.0}]
        evidence = instance.training_evidence("RBD", "complex_cnn_range", Path(self.checkpoint["best_checkpoint_path"]), history, instance.test_rows(str(self.dataset_dir)))
        exported = json.loads((instance.run / evidence["training_inventory_path"]).read_text())
        self.assertEqual(exported["pack"], "data_scale_pack")
        self.assertEqual(len(exported["models"]), 1)
        record = exported["models"][0]
        self.assertEqual(record["training_inventory"], self.checkpoint["training_inventory"])
        self.assertEqual(record["test"]["count"], 5)
        self.assertEqual(record["test"]["source_segment_indices"], [151, 153, 154, 156, 157])
        self.assertEqual(evidence["train_identity_sha256"], record["training_inventory"]["train"]["identity_sha256"])
        self.assertEqual(evidence["test_identity_sha256"], record["test"]["identity_sha256"])
        self.assertEqual(evidence["best_epoch"], 1)
        self.assertEqual(evidence["optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
