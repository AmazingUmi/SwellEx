"""Joint GRNN fold isolation, persistence, prediction and rejection contracts."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.joint_feature_pack.data_processing.joint_features import build_joint_datasets
from scripts.joint_feature_pack.localization_networks.grnn.cli import (
    build_parser, validate_arguments, build_command, predict_command, load_artifact,
    select_sigma_by_cv, select_sigma_by_validation, metric_summary,
)
from scripts.joint_feature_pack.localization_networks.grnn.data import load_grnn_features
from scripts.joint_feature_pack.localization_networks.grnn.model import GrnnRegressor
from scripts.joint_feature_pack.localization_networks.joint.contract import METHODS
from scripts.joint_feature_pack.localization_networks.joint.data import fit_flat_preprocessing, apply_flat_preprocessing
from scripts.joint_feature_pack.localization_networks.joint.tests.fixtures import write_routes


class JointGrnnTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.routes = write_routes(self.root / "routes")
        self.outputs = build_joint_datasets(self.routes, self.root / "joint")

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, method="RBD_ELM_SCM", split="train"):
        return load_grnn_features([Path(self.outputs[method]) / f"{method}_{split}.h5"], method=method)

    def test_cv_scales_are_fitted_per_fold_and_distance_reuse_is_preserved(self):
        bundle = self.bundle()
        bundle.x[-1] *= 1e5
        candidates = [.1, 1., 10.]
        folds = torch.tensor_split(torch.randperm(8, generator=torch.Generator().manual_seed(2026)), 3)
        original_fit = GrnnRegressor.fit_reference
        fitted = []
        def record_fit(model, x, y):
            fitted.append(x.clone())
            return original_fit(model, x, y)
        with patch.object(GrnnRegressor, "fit_reference", record_fit), patch(
                "scripts.joint_feature_pack.localization_networks.grnn.model.torch.cdist", wraps=torch.cdist) as distance:
            with contextlib.redirect_stdout(io.StringIO()):
                _, actual = select_sigma_by_cv(bundle, candidates=candidates, cv_folds=3, seed=2026,
                                               standardize_input=False, batch_size=2, device=torch.device("cpu"))
            self.assertEqual(distance.call_count, 5)
        self.assertEqual(len(fitted), 3)
        expected_metrics = [[] for _ in candidates]
        for fold_id, val_idx in enumerate(folds):
            train_idx = torch.cat([folds[i] for i in range(3) if i != fold_id])
            scales = fit_flat_preprocessing(bundle.x[train_idx], bundle.input_signature)
            train_x = apply_flat_preprocessing(bundle.x[train_idx], bundle.input_signature, scales)
            val_x = apply_flat_preprocessing(bundle.x[val_idx], bundle.input_signature, scales)
            torch.testing.assert_close(fitted[fold_id], train_x, rtol=0, atol=0)
            for index, sigma in enumerate(candidates):
                independent = GrnnRegressor(spread=sigma)
                independent.fit_reference(train_x, bundle.y_range_km[train_idx])
                expected_metrics[index].append(metric_summary(bundle.y_range_km[val_idx], independent.predict(val_x, batch_size=2)))
        for row, metrics in zip(actual, expected_metrics):
            for key in ("mae_km", "rmse_km", "mape_pct"):
                self.assertAlmostEqual(row[f"mean_{key}"], sum(value[key] for value in metrics) / 3, places=7)

    def test_validation_uses_reference_scales_and_rejects_overlapping_identities(self):
        train, validation = self.bundle(), self.bundle(split="test")
        validation.x *= 1e5
        state = fit_flat_preprocessing(train.x, train.input_signature)
        model = GrnnRegressor(spread=1.)
        model.fit_reference(apply_flat_preprocessing(train.x, train.input_signature, state), train.y_range_km)
        expected = metric_summary(validation.y_range_km, model.predict(apply_flat_preprocessing(validation.x, validation.input_signature, state)))
        with contextlib.redirect_stdout(io.StringIO()):
            _, rows = select_sigma_by_validation(train, validation, candidates=[1.], standardize_input=False,
                                                 batch_size=2, device=torch.device("cpu"))
        for key, value in expected.items():
            self.assertAlmostEqual(rows[0][key], value, places=7)
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_sigma_by_validation(train, train, candidates=[1.], standardize_input=False,
                                       batch_size=2, device=torch.device("cpu"))

    def test_every_combination_builds_predicts_and_rejects_cross_pack_artifacts(self):
        parser = build_parser()
        for method in METHODS:
            output = self.root / "models" / method
            args = parser.parse_args(["build", "--method", method, "--data", self.outputs[method],
                                      "--output-dir", str(output), "--cv-folds", "2",
                                      "--sigma-candidates", ".1", "1", "--device", "cpu"])
            validate_arguments(args, parser)
            with contextlib.redirect_stdout(io.StringIO()):
                build_command(args)
            reference = next(output.rglob("*_reference_*.pt"))
            artifact = load_artifact(reference, torch.device("cpu"), method=method)
            self.assertEqual(artifact["input_preprocessing"]["fit_sample_count"], 8)
            self.assertFalse(artifact["standardize_input"])
            bundle = self.bundle(method)
            expected = fit_flat_preprocessing(bundle.x, bundle.input_signature)
            self.assertEqual(artifact["input_preprocessing"], expected)
            restored = GrnnRegressor.from_artifact(artifact)
            torch.testing.assert_close(restored.feature_std, torch.ones_like(restored.feature_std))
            prediction_args = parser.parse_args(["predict", "--method", method, "--data", self.outputs[method],
                                                 "--reference", str(reference), "--output-dir", str(output),
                                                 "--no-plot", "--print-first", "0", "--device", "cpu"])
            validate_arguments(prediction_args, parser)
            with contextlib.redirect_stdout(io.StringIO()):
                predict_command(prediction_args)
            self.assertEqual(len(list(output.rglob("*_predictions_*.csv"))), 1)
            bad = deepcopy(artifact)
            bad["pack"] = "base_pack"
            bad_path = self.root / "bad.pt"
            torch.save(bad, bad_path)
            with self.assertRaisesRegex(ValueError, "pack"):
                load_artifact(bad_path, torch.device("cpu"), method=method)
            bad = deepcopy(artifact)
            bad["input_preprocessing"]["scales"][0] = 0.
            torch.save(bad, bad_path)
            with self.assertRaisesRegex(ValueError, "positive"):
                load_artifact(bad_path, torch.device("cpu"), method=method)

    def test_limits_extra_normalization_and_corrupt_references_reject(self):
        parser = build_parser()
        for options in (["--standardize-input"], ["--per-sample-channel-norm"], ["--cv-folds", "1"],
                        ["--spread", "nan"], ["--batch-size", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args = parser.parse_args(["build", "--method", "RBD", "--data", "fixture", *options])
                validate_arguments(args, parser)
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                load_grnn_features([], method="RBD", max_samples=limit)
        model = GrnnRegressor(spread=1e-8)
        model.fit_reference(torch.tensor([[0., 1.], [2., 3.]]), torch.tensor([[2.], [4.]]))
        artifact = model.to_artifact()
        torch.testing.assert_close(GrnnRegressor.from_artifact(artifact).predict(torch.tensor([[0., 1.]])), torch.tensor([[2.]]))
        for key, replacement in (("feature_std", torch.zeros(2)), ("feature_mean", torch.zeros(3)),
                                 ("x_reference", torch.full((2, 2), float("nan")))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GrnnRegressor.from_artifact(artifact | {key: replacement})


if __name__ == "__main__":
    unittest.main()
