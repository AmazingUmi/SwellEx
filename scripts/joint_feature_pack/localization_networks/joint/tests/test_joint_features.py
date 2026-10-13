"""Alignment, train-only RMS, artifact rejection and existing-architecture checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from scripts.joint_feature_pack.data_processing.joint_features import build_joint_datasets
from scripts.joint_feature_pack.localization_networks.common.h5_utils import split_indices
from scripts.joint_feature_pack.localization_networks.joint.contract import METHODS, validate_signature, validate_preprocessing
from scripts.joint_feature_pack.localization_networks.joint.data import (
    JointRangeH5Dataset, random_split_bundle, fit_flat_preprocessing, apply_flat_preprocessing,
)
from scripts.joint_feature_pack.localization_networks.joint import model
from scripts.joint_feature_pack.localization_networks.joint.tests.fixtures import write_routes


class JointFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.routes = write_routes(self.root / "routes")

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, strategy="block_rms"):
        return build_joint_datasets(self.routes, self.root / strategy, fusion_strategy=strategy)

    def test_all_seven_combinations_preserve_blocks_and_have_no_fake_axis(self):
        outputs = self.build()
        self.assertEqual(set(outputs), set(METHODS))
        for method, directory in outputs.items():
            dataset = JointRangeH5Dataset([Path(directory) / f"{method}_train.h5"], normalize_input=False, method=method)
            try:
                self.assertEqual(len(dataset), 8)
                self.assertNotIn("pair_indices", dataset.input_signature)
                x, _ = dataset[0]
                for block in dataset.input_signature["blocks"]:
                    with h5py.File(self.routes[block["method"]] / f"{block['method']}_train.h5") as handle:
                        np.testing.assert_array_equal(x[:, block["start"]:block["stop"]].numpy(), handle["X"][0].transpose(2, 0, 1))
            finally:
                dataset.close()

    def test_mismatched_rows_windows_frequencies_and_raw_provenance_reject(self):
        for field in ("y_range_km", "time/window_center_s", "split/source_segment_idx", "frequency/freq_hz"):
            path = self.routes["ELM"] / "ELM_train.h5"
            with h5py.File(path, "r+") as handle:
                original = handle[field][0, 0]
                handle[field][0, 0] = original + .25 if "source_segment" not in field else original + 100
            try:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.build()
            finally:
                with h5py.File(path, "r+") as handle:
                    handle[field][0, 0] = original
        meta_path = self.routes["ELM"] / "ELM_metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata["input_files"][0]["sha256"] = "b" * 64
        meta_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.build()

    def test_ns_greater_than_one_rejected_even_if_routes_agree(self):
        for method, directory in self.routes.items():
            path = directory / f"{method}_metadata.json"
            value = json.loads(path.read_text())
            value["effective_config"]["dataset"]["num_snapshots_per_segment"] = 2
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Ns=1"):
            self.build()

    def test_rms_fits_only_actual_training_indices_and_preserves_phase(self):
        outputs = self.build()
        method = "RBD_ELM_SCM"
        path = Path(outputs[method]) / f"{method}_train.h5"
        train_idx, val_idx = split_indices(8, .25, 2026)
        with h5py.File(path, "r+") as handle:
            handle["X"][val_idx[0]] *= 1e6
        bundle = random_split_bundle([path], method=method, normalize_input=False, val_fraction=.25,
                                     seed=2026, split_mode="test")
        dataset = bundle.dataset
        try:
            raw_train = torch.stack([torch.from_numpy(dataset.raw_array(i).copy()).reshape(-1) for i in train_idx])
            expected = fit_flat_preprocessing(raw_train, dataset.input_signature)
            np.testing.assert_allclose(bundle.input_preprocessing["scales"], expected["scales"], rtol=1e-12)
            self.assertEqual(bundle.input_preprocessing["fit_sample_count"], 6)
            all_raw = torch.stack([torch.from_numpy(dataset.raw_array(i).copy()).reshape(-1) for i in range(8)])
            leaked = fit_flat_preprocessing(all_raw, dataset.input_signature)
            self.assertTrue(all(b > a * 100 for a, b in zip(expected["scales"], leaked["scales"])))
            scaled = apply_flat_preprocessing(raw_train, dataset.input_signature, expected).reshape(-1, *dataset.input_shape)
            for block in dataset.input_signature["blocks"]:
                raw = raw_train.reshape(-1, *dataset.input_shape)[:, :, block["start"]:block["stop"]]
                value = scaled[:, :, block["start"]:block["stop"]]
                torch.testing.assert_close(value[:, 0] / value[:, 1], raw[:, 0] / raw[:, 1])
                self.assertAlmostEqual(float(value.double().square().sum(dim=1).mean()), 1.0, places=6)
        finally:
            dataset.close()

    def test_raw_concat_and_malformed_contracts(self):
        outputs = self.build("raw_concat")
        method = "RBD_ELM"
        path = Path(outputs[method]) / f"{method}_train.h5"
        dataset = JointRangeH5Dataset([path], normalize_input=False)
        try:
            raw = torch.stack([dataset[i][0].reshape(-1) for i in range(8)])
            state = fit_flat_preprocessing(raw, dataset.input_signature)
            self.assertEqual(state["scales"], [1., 1.])
            torch.testing.assert_close(apply_flat_preprocessing(raw, dataset.input_signature, state), raw, rtol=0, atol=0)
            broken = deepcopy(dataset.input_signature)
            broken["blocks"][1]["start"] += 1
            with self.assertRaisesRegex(ValueError, "contiguous"):
                validate_signature(broken, dataset.input_shape)
            for scales in ([0., 1.], [float("nan"), 1.], [1.], [2., 1.]):
                with self.subTest(scales=scales), self.assertRaises(ValueError):
                    validate_preprocessing(state | {"scales": scales}, dataset.input_signature)
        finally:
            dataset.close()
        with self.assertRaisesRegex(ValueError, "joint_feature_pack"):
            JointRangeH5Dataset([self.routes["RBD"] / "RBD_train.h5"])

    def test_rms_fitting_closes_handles_before_worker_processes_start(self):
        from torch.utils.data import DataLoader
        method = "RBD_ELM"
        outputs = self.build()
        path = Path(outputs[method]) / f"{method}_train.h5"
        bundle = random_split_bundle([path], method=method, normalize_input=False, val_fraction=.25,
                                     seed=2026, split_mode="worker-test")
        self.assertEqual(bundle.dataset._handles, {})
        loader = DataLoader(bundle.train_dataset, batch_size=2, num_workers=1, shuffle=False)
        batches = list(loader)
        self.assertEqual(sum(len(y) for _, y in batches), len(bundle.train_idx))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x, _ in batches))
        self.assertEqual(bundle.dataset._handles, {})

    def test_existing_architectures_accept_joint_shapes_without_parameter_growth(self):
        from scripts.joint_feature_pack.localization_networks.rbd.network import model as original
        for name in model.MODEL_REGISTRY:
            original_name = name.removeprefix("joint_")
            joint = model.build_model(name, model.build_model_config(name, (2, 12, 4), base_channels=2))
            baseline = original.build_model(original_name, original.build_model_config(original_name, (2, 3, 4), base_channels=2))
            self.assertEqual(sum(p.numel() for p in joint.parameters()), sum(p.numel() for p in baseline.parameters()))
            del baseline
            result = joint(torch.randn(2, 2, 12, 4))
            self.assertEqual(tuple(result.shape), (2, 1))
            result.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in joint.parameters() if p.grad is not None))
            self.assertIn("input_features", model.serialize_model_config(model.build_model_config(name, (2, 12, 4))))


if __name__ == "__main__":
    unittest.main()
