"""SCM-GRNN rejection and persistence contracts; execute on haiqin."""
from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.base_pack.localization_networks.grnn.cli import (
    build_parser, load_artifact, select_sigma_by_validation, validate_arguments,
)
from scripts.base_pack.localization_networks.grnn.data import load_grnn_features
from scripts.base_pack.localization_networks.grnn.model import GrnnRegressor


def write_scm(path: Path, *, reversed_storage: bool = False, frequencies=(10.0, 20.0)) -> None:
    x = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2) / 100
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("X", data=x.transpose(3, 2, 1, 0) if reversed_storage else x)
        dataset.attrs["axis_order"] = "sample,pair,frequency,real_imag"
        dataset.attrs["storage_axis_order"] = (
            "real_imag,frequency,pair,sample" if reversed_storage else "sample,pair,frequency,real_imag"
        )
        for key, value in {
            "y_range_km": [1.0, 2.0, 3.0, 4.0], "valid_sample": [1, 1, 0, 1],
            "frequency/freq_hz": frequencies, "array/element_idx": [1, 2],
            "array/depth_m": [10.0, 20.0], "pair/numerator_element_idx": [1, 1, 2],
            "pair/denominator_element_idx": [1, 2, 2],
            "split/source_segment_idx": [11, 12, 13, 14], "time/window_center_s": [0.5, 1.5, 2.5, 3.5],
        }.items():
            handle.create_dataset(key, data=np.asarray(value).reshape(-1, 1))


class GrnnContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.native = self.directory / "native.h5"
        write_scm(self.native)

    def tearDown(self):
        self.temporary.cleanup()

    def test_storage_axes_and_sample_identity(self):
        reverse = self.directory / "reverse.h5"
        write_scm(reverse, reversed_storage=True)
        native = load_grnn_features([self.native], method="SCM")
        reversed_bundle = load_grnn_features([reverse], method="SCM")
        torch.testing.assert_close(native.x, reversed_bundle.x)
        self.assertEqual(native.input_signature, reversed_bundle.input_signature)
        self.assertEqual(native.source_segment_idx, [11, 12, 14])
        self.assertEqual(native.window_center_s, [0.5, 1.5, 3.5])
        self.assertEqual(native.input_shape, (2, 3, 2))

    def test_same_shape_changed_frequency_is_rejected(self):
        changed = self.directory / "changed.h5"
        write_scm(changed, frequencies=(11.0, 20.0))
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            load_grnn_features([self.native, changed], method="SCM")
        train, val = (load_grnn_features([path], method="SCM") for path in (self.native, changed))
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            select_sigma_by_validation(train, val, candidates=[0.1], standardize_input=False,
                                       batch_size=2, device=torch.device("cpu"))

    def test_bad_features_masks_and_pair_order_are_rejected(self):
        for field in ("X", "valid_sample", "pair/numerator_element_idx"):
            with self.subTest(field=field):
                path = self.directory / f"bad_{field.replace('/', '_')}.h5"
                write_scm(path)
                with h5py.File(path, "r+") as handle:
                    if field == "X":
                        handle[field][0, 0, 0, 0] = float("nan")
                    elif field == "valid_sample":
                        handle[field][0, 0] = 2
                    else:
                        handle[field][1, 0] = 2
                with self.assertRaises(ValueError):
                    load_grnn_features([path], method="SCM")
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_grnn_features([self.native], method="SCM", max_samples=limit)

    def test_small_sigma_and_persistence_preserve_finite_predictions(self):
        bundle = load_grnn_features([self.native], method="SCM")
        model = GrnnRegressor(spread=1e-8, standardize_input=True)
        model.fit_reference(bundle.x, bundle.y_range_km)
        pred = model.predict(bundle.x, batch_size=1)
        torch.testing.assert_close(pred, bundle.y_range_km)
        restored = GrnnRegressor.from_artifact(model.to_artifact())
        torch.testing.assert_close(restored.predict(bundle.x, batch_size=2), pred)

    def test_invalid_model_and_artifact_values_are_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), True):
            for option in ("spread", "eps"):
                with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                    GrnnRegressor(**{option: value})
        model = GrnnRegressor()
        for x, y in ((torch.empty(0, 2), torch.empty(0, 1)),
                     (torch.tensor([[float("inf"), 0]]), torch.ones(1, 1)),
                     (torch.ones(1, 2), torch.ones(1, 2))):
            with self.assertRaises(ValueError):
                model.fit_reference(x, y)
        model.fit_reference(torch.zeros(2, 2), torch.ones(2, 1))
        for batch_size in (0, -1, True):
            with self.assertRaises(ValueError):
                model.predict(torch.zeros(1, 2), batch_size=batch_size)
        for query in (torch.empty(0, 2), torch.full((1, 2), float("nan"))):
            with self.assertRaises(ValueError):
                model.predict(query)
        for name, replacement in (("feature_std", torch.zeros(2)),
                                  ("feature_mean", torch.zeros(3)),
                                  ("y_reference_km", torch.ones(2)),
                                  ("x_reference", torch.full((2, 2), float("nan")))):
            artifact = copy.deepcopy(model.to_artifact())
            artifact[name] = replacement
            with self.subTest(name=name), self.assertRaises(ValueError):
                GrnnRegressor.from_artifact(artifact)

    def test_extreme_sigma_and_eps_are_rejected_before_prediction(self):
        for spread in (1e-300, 1e300):
            with self.subTest(spread=spread), self.assertRaisesRegex(ValueError, "float32"):
                GrnnRegressor(spread=spread)
        with self.assertRaisesRegex(ValueError, "kernel weights"):
            GrnnRegressor(eps=1.01)
        model = GrnnRegressor(spread=0.01, eps=1.0)
        model.fit_reference(torch.tensor([[0.0], [1.0]]), torch.tensor([[2.0], [8.0]]))
        pred = model.predict(torch.tensor([[0.0], [0.5], [1.0]]))
        torch.testing.assert_close(pred, torch.tensor([[2.0], [5.0], [8.0]]))

    def test_reference_contract_requires_signature_and_matching_counts(self):
        bundle = load_grnn_features([self.native], method="SCM")
        model = GrnnRegressor()
        model.fit_reference(bundle.x, bundle.y_range_km)
        artifact = dict(model.to_artifact(), artifact_type="scm_grnn_reference", method="scm_grnn_range",
                        input_shape=bundle.input_shape, input_signature=bundle.input_signature,
                        feature_dim=12, reference_sample_count=3, per_sample_channel_norm=False)
        path = self.directory / "reference.pt"
        torch.save(artifact, path)
        self.assertEqual(load_artifact(path, torch.device("cpu"), method="SCM")["input_signature"], bundle.input_signature)
        for name, value in (("reference_sample_count", 4), ("feature_dim", 13), ("input_signature", None)):
            bad = artifact | {name: value}
            torch.save(bad, path)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_artifact(path, torch.device("cpu"), method="SCM")

    def test_cli_rejects_invalid_limits_and_sigmas(self):
        for argv in (("build", "--spread", "nan"), ("build", "--spread", "1e-300"),
                     ("build", "--seed", "-1"), ("build", "--seed", "4294967296"),
                     ("predict", "--batch-s", "2"), ("build", "--cv-folds", "1"),
                     ("build", "--sigma-candidates", "inf"), ("predict", "--batch-size", "0"),
                     ("predict", "--max-predict-samples", "-1"), ("predict", "--print-first", "-1")):
            parser = build_parser()
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                args = parser.parse_args([argv[0], "--method", "SCM", "--data", "fixture", *argv[1:]])
                validate_arguments(args, parser)


if __name__ == "__main__":
    unittest.main()
