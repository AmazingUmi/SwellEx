"""Persist train-only preprocessing and refuse mismatched continuation."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import h5py

from scripts.joint_feature_pack.data_processing.joint_features import build_joint_datasets
from scripts.joint_feature_pack.localization_networks.common.checkpoints import validate_checkpoint
from scripts.joint_feature_pack.localization_networks.common.cli import build_parser
from scripts.joint_feature_pack.localization_networks.common.prediction import predict
from scripts.joint_feature_pack.localization_networks.common.training import train
from scripts.joint_feature_pack.localization_networks.joint.bindings import make_bindings
from scripts.joint_feature_pack.localization_networks.joint.tests.fixtures import write_routes


class JointArtifactTests(unittest.TestCase):
    def test_train_predict_and_resume_require_the_same_fitted_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = write_routes(root / "routes")
            data = build_joint_datasets(sources, root / "joint")["RBD_ELM_SCM"]
            bindings = make_bindings("RBD_ELM_SCM")
            parser = build_parser(bindings)
            model_name = "joint_complex_cnn_range"
            shared = ["--data", data, "--model", model_name, "--device", "cpu", "--output-dir", str(root / "models")]
            args = parser.parse_args(["train", *shared, "--epochs", "1", "--batch-size", "2",
                                      "--base-channels", "2", "--no-resume"])
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    "scripts.joint_feature_pack.localization_networks.common.training.time_suffix", return_value="0101_000000"):
                train(args, bindings)
            last = next((root / "models").rglob("*_last_*.pt"))
            saved = torch.load(last, map_location="cpu", weights_only=True)
            self.assertEqual(saved["pack"], "joint_feature_pack")
            self.assertEqual(saved["input_preprocessing"]["fit_sample_count"], len(saved["train_indices"]))
            self.assertEqual(saved["model_config"]["input_features"], 12)
            self.assertNotIn("input_elements", saved["model_config"])
            training_path = Path(data) / "RBD_ELM_SCM_train.h5"
            expected_source = {"path": str(training_path.resolve()), "size_bytes": training_path.stat().st_size,
                               "sha256": hashlib.sha256(training_path.read_bytes()).hexdigest()}
            self.assertEqual(saved["resume_data_identity"], {"format_version": 1,
                             "train_sources": [expected_source], "val_sources": [expected_source]})
            prediction = parser.parse_args(["predict", *shared, "--checkpoint", str(last), "--no-plot", "--print-first", "0"])
            with contextlib.redirect_stdout(io.StringIO()):
                predict(prediction, bindings)
            csv = next((root / "models").rglob("*_predictions_*.csv"))
            self.assertEqual(len(csv.read_text().splitlines()), 4)
            changed = deepcopy(saved)
            changed["input_preprocessing"]["scales"][0] *= 2
            bad_path = root / "changed.pt"
            torch.save(changed, bad_path)
            resume = parser.parse_args(["train", *shared, "--epochs", "1", "--batch-size", "2",
                                        "--resume-checkpoint", str(bad_path)])
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                train(resume, bindings)
            for field, value in (("pack", "base_pack"), ("input_preprocessing", None), ("resume_data_identity", None)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_checkpoint(saved | {field: value}, method=bindings.method,
                                        model_name=model_name, dataset_name=data)

    def test_raw_concat_resume_rejects_reordered_rows_changed_features_and_wrong_historical_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = write_routes(root / "routes")
            data = build_joint_datasets(sources, root / "joint", fusion_strategy="raw_concat")["RBD_ELM_SCM"]
            bindings = make_bindings("RBD_ELM_SCM")
            parser = build_parser(bindings)
            shared = ["--data", data, "--model", "joint_complex_cnn_range", "--device", "cpu",
                      "--output-dir", str(root / "models"), "--epochs", "1", "--batch-size", "2", "--base-channels", "2"]
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    "scripts.joint_feature_pack.localization_networks.common.training.time_suffix", return_value="0101_000000"):
                train(parser.parse_args(["train", *shared, "--no-resume"]), bindings)
            last_path = next((root / "models").rglob("*_last_*.pt"))
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["input_preprocessing"]["scales"], [1., 1., 1.])
            training_path = Path(data) / "RBD_ELM_SCM_train.h5"
            original_bytes = training_path.read_bytes()
            resume_args = parser.parse_args(["train", *shared, "--resume-checkpoint", str(last_path)])
            for mutation in ("row_order", "same_id_features"):
                with self.subTest(mutation=mutation):
                    with h5py.File(training_path, "r+") as handle:
                        if mutation == "row_order":
                            datasets = []
                            handle.visititems(lambda _, value: datasets.append(value) if isinstance(value, h5py.Dataset) else None)
                            for dataset in datasets:
                                dataset[...] = dataset[...][::-1]
                        else:
                            handle["X"][0, 0, 0, 0] += .5
                    with self.assertRaisesRegex(ValueError, "source content identity differs"):
                        train(resume_args, bindings)
                    training_path.write_bytes(original_bytes)
            # Current last checkpoint/data agree, but an old best from other
            # source bytes must not be silently retained during continuation.
            altered_best = deepcopy(checkpoint)
            altered_best["resume_data_identity"]["train_sources"][0]["sha256"] = "b" * 64
            bad_best_path = root / "other_best.pt"
            torch.save(altered_best, bad_best_path)
            resume_checkpoint = deepcopy(checkpoint)
            resume_checkpoint["best_checkpoint_path"] = str(bad_best_path)
            altered_last_path = root / "last_with_other_best.pt"
            torch.save(resume_checkpoint, altered_last_path)
            resume_args.resume_checkpoint = altered_last_path
            with self.assertRaisesRegex(ValueError, "Historical best training source content identity"):
                train(resume_args, bindings)


if __name__ == "__main__":
    unittest.main()
