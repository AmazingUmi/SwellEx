"""Remote-only HDF5 alignment, training-only RMS and CUDA model checks.

Run on haiqin with SWELLEX_MOE_TEST_DEVICE=cuda; never fall back to CPU.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.MoE_data_scale.data_processing.joint_features import build_joint_dataset
from scripts.MoE_data_scale.localization_networks.common.checkpoints import validate_checkpoint
from scripts.MoE_data_scale.localization_networks.common.data_selection import select_training_subset
from scripts.MoE_data_scale.localization_networks.common.scaling import DEFAULT_SCALING_POLICY
from scripts.MoE_data_scale.localization_networks.common.training_utils import controlled_sampler_plan
from scripts.MoE_data_scale.localization_networks.rbd_scm.network import data, model
from .test_contract import joint_signature

REQUESTED_DEVICE = None


def native_datasets(root):
    """Two native feature routes sharing exact original-input/window identities."""
    result = {}
    signature = joint_signature()
    for route, rows in (("RBD", 2), ("SCM", 3)):
        directory = root / route
        directory.mkdir()
        result[route] = str(directory)
        metadata = dict(signature["source_contract"])
        (directory / f"{route}_metadata.json").write_text(json.dumps(metadata))
        for split, ids in (("train", np.arange(1, 13)), ("test", np.arange(13, 17))):
            with h5py.File(directory / f"{route}_{split}.h5", "w") as handle:
                n = len(ids)
                factor = 1. if route == "RBD" else 10.
                features = np.broadcast_to(ids[:, None, None, None] * factor, (n, rows, 3, 2)).astype(np.float32)
                x = handle.create_dataset("X", data=features)
                order = f"sample,{'element' if route == 'RBD' else 'pair'},frequency,real_imag"
                x.attrs["axis_order"] = x.attrs["storage_axis_order"] = order
                handle.attrs["generator_version"] = "test-v1"
                for key, values in {"y_range_km": ids / 10., "valid_sample": np.ones(n), "split/source_segment_idx": ids,
                                    "time/window_start_s": ids - 1., "time/window_center_s": ids - .5,
                                    "time/window_stop_s": ids.astype(float), "time/sample_start_idx": ids * 100 - 99,
                                    "time/sample_stop_idx": ids * 100, "frequency/freq_hz": [100., 105., 110.],
                                    "array/element_idx": [1, 2], "array/depth_m": [100., 110.],
                                    "range/range_time_s": np.arange(16), "range/range_km_raw": np.arange(16) / 10.}.items():
                    handle.create_dataset(key, data=np.asarray(values).reshape(-1, 1))
                if route == "SCM":
                    handle.create_dataset("pair/numerator_element_idx", data=[1, 1, 2])
                    handle.create_dataset("pair/denominator_element_idx", data=[1, 2, 2])
    return result


class JointInputContracts(unittest.TestCase):
    def setUp(self):
        requested = REQUESTED_DEVICE or os.environ.get("SWELLEX_MOE_TEST_DEVICE")
        if not requested or torch.device(requested).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Joint numerical checks require explicitly selected CUDA on haiqin")
        self.device = torch.device(requested)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dirs = native_datasets(self.root)

    def build(self):
        return Path(build_joint_dataset(self.dirs, self.root / "joint")["RBD_SCM"])

    def bundle(self):
        directory = self.build()
        return data.random_split_bundle([directory / "RBD_SCM_train.h5"], normalize_input=False,
                                        val_fraction=.25, seed=2026, split_mode="test")

    def test_native_join_preserves_raw_order_and_rejects_unfitted_access(self):
        directory = self.build()
        dataset = data.JointRangeH5Dataset([directory / "RBD_SCM_train.h5"], normalize_input=False)
        self.addCleanup(dataset.close)
        self.assertEqual(dataset.input_shape, (2, 5, 3))
        self.assertEqual(dataset.input_signature["method"], "RBD_SCM")
        np.testing.assert_allclose(dataset.raw_array(0)[:, :2], 1.)
        np.testing.assert_allclose(dataset.raw_array(0)[:, 2:], 10.)
        with self.assertRaisesRegex(ValueError, "fitted or restored"):
            dataset[0]
        with self.assertRaises(FileExistsError):
            build_joint_dataset(self.dirs, self.root / "joint")

    def test_join_rejects_sample_frequency_pair_source_and_ns_mismatch(self):
        cases = ("labels", "frequency", "pairs", "source", "Ns")
        for case in cases:
            with self.subTest(case=case):
                case_root = self.root / case
                case_root.mkdir()
                dirs = native_datasets(case_root)
                if case in ("source", "Ns"):
                    path = Path(dirs["SCM"]) / "SCM_metadata.json"
                    metadata = json.loads(path.read_text())
                    if case == "source":
                        metadata["input_files"][0]["sha256"] = "b" * 64
                    else:
                        metadata["effective_config"]["dataset"]["num_snapshots_per_segment"] = 2
                    path.write_text(json.dumps(metadata))
                else:
                    with h5py.File(Path(dirs["SCM"]) / "SCM_train.h5", "r+") as handle:
                        key = {"labels": "y_range_km", "frequency": "frequency/freq_hz", "pairs": "pair/numerator_element_idx"}[case]
                        handle[key][0] = 2 if case == "pairs" else 101
                with self.assertRaises(ValueError):
                    build_joint_dataset(dirs, case_root / "joint")

    def test_rms_fits_only_retained_training_members_and_resume_never_refits(self):
        bundle, sampling = select_training_subset(self.bundle(), .5, 2026)
        self.addCleanup(bundle.dataset.close)
        self.assertIsNone(bundle.dataset.preprocessing_state, "Bundle construction must never fit preprocessing")
        retained = bundle.train_idx
        excluded = sorted(set(range(len(bundle.dataset))) - set(retained))
        path = bundle.dataset.h5_paths[0]
        # If either discarded training candidates or validation rows enter RMS,
        # these deliberately huge values visibly corrupt the expected scale.
        with h5py.File(path, "r+") as handle:
            for index in excluded:
                handle["X"][index] = 1.e9
        state = data.prepare_preprocessing(bundle, sampling)
        expected = math.sqrt(2. * sum((i + 1.) ** 2 for i in retained) / len(retained))
        self.assertAlmostEqual(state["scales"][0], expected)
        self.assertAlmostEqual(state["scales"][1], 10 * expected)
        self.assertEqual(state["fit_sample_count"], len(retained))
        self.assertEqual(state["fit_source_identity_sha256"], sampling["source_identity_sha256"])
        self.assertIs(bundle.val_dataset.dataset, bundle.dataset)
        with patch.object(data, "fit_preprocessing", side_effect=AssertionError("resume refit")):
            self.assertEqual(data.prepare_preprocessing(bundle, sampling, saved_state=state), state)
        bad = dict(state, fit_source_identity_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "membership"):
            data.prepare_preprocessing(bundle, sampling, saved_state=bad)
        directory = path.parent
        test = data.JointRangeH5Dataset([directory / "RBD_SCM_test.h5"], normalize_input=False,
                                       preprocessing_state=state)
        self.addCleanup(test.close)
        x, _ = test[0]
        torch.testing.assert_close(x[:, :2], torch.full_like(x[:, :2], 13 / expected))
        torch.testing.assert_close(x[:, 2:], torch.full_like(x[:, 2:], 13 / expected))
        self.assertEqual(test.preprocessing_state, state, "Held-out data must reuse training scales unchanged")

    def test_joint_models_checkpoint_identity_and_cuda_forward_backward(self):
        signature = joint_signature()
        digest = hashlib.sha256(b"1,2,3,4").hexdigest()
        state = {"strategy": "block_rms", "scales": [1., 10.], "fit_sample_count": 4,
                 "fit_source_identity_sha256": digest}
        for selection in ("range", "e1_range", "e2_range", "e4_range", "e8_range"):
            name = f"rbd_scm_complex_moe_{selection}"
            config = model.build_model_config(name, (2, 5, 3), base_channels=2, dropout=0., train_count=4,
                                              scaling_policy=dict(DEFAULT_SCALING_POLICY))
            network = model.build_model(name, config).to(self.device)
            x = torch.randn(3, 2, 5, 3, device=self.device, requires_grad=True)
            predictions, aux = network.forward_with_aux(x)
            self.assertEqual(tuple(predictions.shape), (3, 1))
            self.assertTrue(bool(torch.isfinite(predictions).all()))
            (predictions.square().mean() + .01 * aux["balance_loss"]).backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertEqual(int(aux["expert_load"].sum()), 3 * config.top_k)
            self.assertEqual(sum(p.numel() for p in network.parameters()), config.parameter_count)
            checkpoint = {"checkpoint_version": 1, "pack": "MoE_data_scale", "method": "RBD_SCM",
                          "model_name": name, "dataset_name": "fixture", "model_state": network.state_dict(),
                          "model_config": model.serialize_model_config(config), "scaling": model.resolution_metadata(config),
                          "sampling": {"train_count": 4, "train_fraction": .5, "source_identity_sha256": digest},
                          "source_identity": {}, "train_fraction": .5, "input_signature": signature,
                          "input_normalized_per_sample": False, "input_preprocessing": state,
                          "target_mean_km": 1., "target_std_km": 1., "loss_space": "km", "huber_beta": 1.,
                          "balance_loss_weight": .01}
            validate_checkpoint(checkpoint, method="RBD_SCM", model_name=name, dataset_name="fixture")
            bad = dict(checkpoint)
            del bad["input_preprocessing"]
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, method="RBD_SCM", model_name=name, dataset_name="fixture")
            bad = dict(checkpoint, input_preprocessing=dict(state, fit_sample_count=5))
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, method="RBD_SCM", model_name=name, dataset_name="fixture")

    def test_independent_epoch_generators_preserve_batches_across_expert_counts(self):
        batches = []
        for experts in (1, 8):
            seed = 2026
            torch.manual_seed(seed)
            sampler = controlled_sampler_plan(seed)
            generator = torch.Generator().manual_seed(sampler["train_seed"])
            loader = DataLoader(TensorDataset(torch.arange(20)), batch_size=6, shuffle=True, generator=generator)
            name = f"rbd_scm_complex_moe_e{experts}_range"
            config = model.build_model_config(name, (2, 5, 3), base_channels=2, dropout=0., train_count=20,
                                              scaling_policy=dict(DEFAULT_SCALING_POLICY))
            model.build_model(name, config).to(self.device)
            # Initialization and validation can consume global RNG arbitrarily.
            torch.randn(experts * 100, device=self.device)
            batches.append([batch[0].tolist() for batch in loader])
        self.assertEqual(batches[0], batches[1])


if __name__ == "__main__":
    unittest.main()
