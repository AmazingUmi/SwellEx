"""Distance reuse must preserve GRNN candidate metrics and fold isolation."""
from __future__ import annotations

import contextlib
import io
import math
import unittest
from unittest import mock

import torch

from scripts.localization_networks.grnn.cli import (
    metric_summary, select_sigma_by_cv, select_sigma_by_validation,
)
from scripts.localization_networks.grnn.data import GrnnFeatureBundle
from scripts.localization_networks.grnn.model import GrnnRegressor


def bundle(count: int, *, offset: float = 0.0) -> GrnnFeatureBundle:
    positions = torch.arange(count * 12, dtype=torch.float32).reshape(count, 12)
    x = torch.sin(positions * 0.23 + offset) + positions / 80.0
    y = (1.0 + torch.arange(count, dtype=torch.float32) / 10.0).reshape(-1, 1)
    return GrnnFeatureBundle(
        x=x, y_range_km=y, source_segment_idx=list(range(1, count + 1)),
        window_center_s=[float(i) + 0.5 for i in range(count)],
        input_shape=(2, 3, 2), source_paths=[],
        input_signature={"input_shape": [2, 3, 2], "frequency_hz": [10.0, 20.0]},
    )


def original_predict(model: GrnnRegressor, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """The previous single-sigma calculation, independent of predict_candidates."""
    reference, labels, mean, std = model._require_reference()
    outputs = []
    for start in range(0, len(x), batch_size):
        query = (x[start:start + batch_size] - mean) / std.clamp_min(model.eps)
        distances = torch.cdist(query, reference, p=2).square()
        distances = distances - distances.min(dim=1, keepdim=True).values
        weights = torch.exp(-distances / model.kernel_scale)
        outputs.append((weights @ labels) / weights.sum(dim=1, keepdim=True).clamp_min(model.eps))
    return torch.cat(outputs).cpu()


def original_cv(data, candidates, folds, seed, standardize, batch_size):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.tensor_split(torch.randperm(len(data.x), generator=generator), folds)
    results = []
    for sigma in candidates:
        metrics = []
        for fold_id in range(folds):
            validation = indices[fold_id]
            training = torch.cat([indices[i] for i in range(folds) if i != fold_id])
            model = GrnnRegressor(spread=sigma, standardize_input=standardize)
            model.fit_reference(data.x[training], data.y_range_km[training])
            pred = original_predict(model, data.x[validation], batch_size)
            metrics.append(metric_summary(data.y_range_km[validation], pred))
        results.append({"sigma": float(sigma), **{
            f"mean_{name}": sum(row[name] for row in metrics) / folds
            for name in ("mae_km", "rmse_km", "mape_pct")
        }})
    return min(results, key=lambda row: row["mean_mape_pct"])["sigma"], results


class SigmaSearchReuse(unittest.TestCase):
    candidates = [0.03, 0.2, 1.3, 2.0]

    def assert_metrics_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual):
            self.assertEqual(before.keys(), after.keys())
            for name in before:
                self.assertTrue(math.isclose(before[name], after[name], rel_tol=1e-6, abs_tol=1e-7),
                                f"{name}: {before[name]} != {after[name]}")

    def test_cv_matches_previous_algorithm_with_fold_fitted_standardization(self):
        data = bundle(19)
        for standardize in (False, True):
            for batch_size in (1, 4, 64):
                with self.subTest(standardize=standardize, batch_size=batch_size):
                    expected_sigma, expected = original_cv(data, self.candidates, 4, 2026,
                                                           standardize, batch_size)
                    with contextlib.redirect_stdout(io.StringIO()):
                        sigma, actual = select_sigma_by_cv(
                            data, candidates=self.candidates, cv_folds=4, seed=2026,
                            standardize_input=standardize, batch_size=batch_size,
                            device=torch.device("cpu"),
                        )
                    self.assertEqual(sigma, expected_sigma)
                    self.assert_metrics_equal(expected, actual)

    def test_validation_matches_previous_algorithm_and_uses_train_statistics(self):
        training, validation = bundle(19), bundle(7, offset=3.0)
        validation.x += 10.0
        for standardize in (False, True):
            for batch_size in (1, 4, 64):
                with self.subTest(standardize=standardize, batch_size=batch_size):
                    expected = []
                    for candidate in self.candidates:
                        model = GrnnRegressor(spread=candidate, standardize_input=standardize)
                        model.fit_reference(training.x, training.y_range_km)
                        expected.append({"sigma": candidate, **metric_summary(
                            validation.y_range_km, original_predict(model, validation.x, batch_size))})
                    with contextlib.redirect_stdout(io.StringIO()):
                        sigma, actual = select_sigma_by_validation(
                            training, validation, candidates=self.candidates,
                            standardize_input=standardize, batch_size=batch_size,
                            device=torch.device("cpu"),
                        )
                    self.assertEqual(sigma, min(expected, key=lambda row: row["mape_pct"])["sigma"])
                    self.assert_metrics_equal(expected, actual)

    def test_distances_are_computed_once_per_query_batch_and_fold(self):
        training, validation = bundle(19), bundle(7)
        fit = GrnnRegressor.fit_reference
        fit_inputs = []

        def record_fit(model, x, y):
            fit_inputs.append(x.clone())
            return fit(model, x, y)

        with mock.patch("scripts.localization_networks.grnn.model.torch.cdist", wraps=torch.cdist) as distances:
            with mock.patch.object(GrnnRegressor, "fit_reference", record_fit):
                with contextlib.redirect_stdout(io.StringIO()):
                    select_sigma_by_cv(training, candidates=self.candidates, cv_folds=4,
                                       seed=2026, standardize_input=True, batch_size=3,
                                       device=torch.device("cpu"))
            self.assertEqual(distances.call_count, 8)
        indices = torch.tensor_split(torch.randperm(19, generator=torch.Generator().manual_seed(2026)), 4)
        self.assertEqual(len(fit_inputs), 4)
        for fold_id, x in enumerate(fit_inputs):
            expected = torch.cat([indices[i] for i in range(4) if i != fold_id])
            torch.testing.assert_close(x, training.x[expected], rtol=0, atol=0)

        fit_inputs.clear()
        with mock.patch("scripts.localization_networks.grnn.model.torch.cdist", wraps=torch.cdist) as distances:
            with mock.patch.object(GrnnRegressor, "fit_reference", record_fit):
                with contextlib.redirect_stdout(io.StringIO()):
                    select_sigma_by_validation(training, validation, candidates=self.candidates,
                                               standardize_input=True, batch_size=3,
                                               device=torch.device("cpu"))
            self.assertEqual(distances.call_count, 3)
        self.assertEqual(len(fit_inputs), 1)
        torch.testing.assert_close(fit_inputs[0], training.x, rtol=0, atol=0)

    def test_candidate_predictions_preserve_order_single_sigma_and_batch_behavior(self):
        training, query = bundle(19), bundle(7, offset=0.5)
        for standardize in (False, True):
            model = GrnnRegressor(spread=0.2, standardize_input=standardize)
            model.fit_reference(training.x, training.y_range_km)
            candidates = [1.3, 0.03, 1.3]
            baseline = None
            for batch_size in (1, 4, 64):
                actual = model.predict_candidates(query.x, candidates=candidates, batch_size=batch_size)
                if baseline is None:
                    baseline = actual
                else:
                    for before, after in zip(baseline, actual):
                        torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
                for sigma, pred in zip(candidates, actual):
                    original = GrnnRegressor(spread=sigma, standardize_input=standardize)
                    original.fit_reference(training.x, training.y_range_km)
                    torch.testing.assert_close(pred, original_predict(original, query.x, batch_size),
                                               rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(model.predict(query.x, batch_size=batch_size),
                                           original_predict(model, query.x, batch_size), rtol=1e-6, atol=1e-7)
            with self.assertRaisesRegex(ValueError, "candidate"):
                model.predict_candidates(query.x, candidates=[])
            with self.assertRaisesRegex(ValueError, "float32"):
                model.predict_candidates(query.x, candidates=[0.2, float("nan")])


if __name__ == "__main__":
    unittest.main()
