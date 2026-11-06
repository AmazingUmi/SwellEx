"""Critical rejection and continuation cases, executed on haiqin only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.network_training.neural.bindings import MethodBindings
from scripts.network_training.neural.checkpoints import validate_checkpoint
from scripts.network_training.common.h5_utils import blocked_split_indices, split_indices
from scripts.data_processing.input_contract import read_input_signature
from scripts.network_training.neural.training import train
from scripts.network_training.neural.training_utils import run_epoch


def fixture(path: Path, method: str, *, frequency_offset=0, reverse=False):
    n, e, f = 4, 3, 5
    pairs = [(i, j) for j in range(1, e + 1) for i in range(1, j + (method == "SCM"))]
    height = e if method == "RBD" else len(pairs)
    values = np.arange(n * height * f * 2, dtype=np.float32).reshape(n, height, f, 2)
    with h5py.File(path, "w") as handle:
        x = handle.create_dataset("X", data=values.transpose(3, 2, 1, 0) if reverse else values)
        axes = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
        x.attrs["axis_order"] = axes
        x.attrs["storage_axis_order"] = ",".join(axes.split(",")[::-1]) if reverse else axes
        handle["y_range_km"] = np.arange(n, dtype=float)
        handle["array/depth_m"] = np.arange(e) + 100.0
        handle["array/element_idx"] = np.arange(e) + 1
        handle["frequency/freq_hz"] = np.arange(f) * 10.0 + frequency_offset
        handle["split/source_segment_idx"] = np.arange(n) + 1
        handle["time/window_center_s"] = np.arange(n, dtype=float)
        if method != "RBD":
            handle["pair/numerator_element_idx"] = [p[0] for p in pairs]
            handle["pair/denominator_element_idx"] = [p[1] for p in pairs]
    return (2, height, f)


class NetworkContractTests(unittest.TestCase):
    def test_invalid_splits_and_empty_epoch_fail(self):
        for count, fraction in [(1, .25), (7, 0), (7, 1), (7, -1), (7, float("nan"))]:
            with self.subTest(count=count, fraction=fraction), self.assertRaises(ValueError):
                split_indices(count, fraction, 2026)
        empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, 1)))
        with self.assertRaisesRegex(ValueError, "empty epoch"):
            run_epoch(nn.Linear(2, 1), empty, nn.SmoothL1Loss(), torch.device("cpu"), 0, 1)

    def test_cli_rejects_invalid_and_conflicting_arguments(self):
        from scripts.network_training.neural.cli import build_parser
        from scripts.network_training.neural.model import models_for
        parser = build_parser()
        for method in ("RBD", "SCM"):
            model = models_for(method)[0]
            for extra in (["--no-resume", "--resume-checkpoint", "x.pt"], ["--epochs", "0"],
                          ["--val-fraction", "0"], ["--val-fraction", "nan"], ["--lr", "nan"],
                          ["--dropout", "1"], ["--num-workers", "-1"], ["--seed", "-1"],
                          ["--seed", str(2**32)], ["--ep", "1"]):
                with self.subTest(method=method, extra=extra), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["train", "--method", method, "--model", model, "--data", "test", *extra])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["train", "--method", "RBD", "--model", models_for("SCM")[0], "--data", "test"])
        self.assertEqual(parser.parse_args(["predict", "--method", "SCM", "--model", models_for("SCM")[0], "--data", "t"]).output_dir.name, "SCM")

    def test_seed_range_matches_public_workflow(self):
        from scripts.network_training.neural.training_utils import seed_everything
        for seed in (-1, 2**32, False, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                seed_everything(seed)
        from scripts.network_training.common.cli_args import seed_int
        self.assertEqual(seed_int("0"), 0)
        self.assertEqual(seed_int(str(2**32 - 1)), 2**32 - 1)

    def test_explicit_missing_prediction_checkpoint_never_uses_timestamp_match(self):
        from scripts.network_training.neural.prediction import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "requested.pt"
            # A genuine matching candidate exists; it must not satisfy an explicit path.
            (root / "requested_0101_000001.pt").write_bytes(b"different checkpoint")
            args = argparse.Namespace(batch_size=1, num_workers=0, print_first=0, data="test",
                                      output_dir=root, model="test_model", checkpoint=missing, device="cpu")
            with patch("scripts.network_training.neural.prediction.latest_time_suffixed_path") as latest, patch("scripts.network_training.neural.prediction.torch.load") as load:
                with self.assertRaisesRegex(FileNotFoundError, "Explicit prediction checkpoint"):
                    predict(args, SimpleNamespace(method="RBD"))
                latest.assert_not_called()
                load.assert_not_called()

    def test_missing_checkpoint_contract_and_wrong_model_fail(self):
        with self.assertRaisesRegex(ValueError, "missing contract fields"):
            validate_checkpoint({"model_name": "complex_cnn_range"}, method="RBD", model_name="complex_cnn_range", dataset_name="test")
        checkpoint = {"checkpoint_version": 1, "method": "RBD", "model_name": "resnet18_range", "dataset_name": "test",
                      "model_config": {}, "model_state": {}, "input_signature": {"method": "RBD"},
                      "input_normalized_per_sample": False, "target_mean_km": 0, "target_std_km": 1,
                      "loss_space": "km", "huber_beta": .5}
        with self.assertRaisesRegex(ValueError, "model_name"):
            validate_checkpoint(checkpoint, method="RBD", model_name="complex_cnn_range", dataset_name="test")

    def test_same_shape_different_frequency_rejected_across_files_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method, cls_name in (("RBD", "RbdRangeH5Dataset"), ("SCM", "ScmRangeH5Dataset")):
                module = importlib.import_module("scripts.network_training.neural.data")
                a, b = Path(tmp) / "a.h5", Path(tmp) / "b.h5"
                fixture(a, method); fixture(b, method, frequency_offset=1)
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "input signature mismatch"):
                    getattr(module, cls_name)([a, b])
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "input signature mismatch"):
                    module.fixed_split_bundle(getattr(module, cls_name), [a], [b], normalize_input=False, split_mode="fixed")

    def test_pair_semantics_missing_metadata_and_nonfinite_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.h5"
            for method in ("SCM",):
                shape = fixture(path, method)
                with h5py.File(path, "r+") as handle:
                    handle["pair/numerator_element_idx"][0] = 2
                    with self.assertRaisesRegex(ValueError, "column-major"):
                        read_input_signature(handle, shape, method)
            shape = fixture(path, "RBD")
            with h5py.File(path, "r+") as handle:
                del handle["frequency/freq_hz"]
                with self.assertRaisesRegex(ValueError, "missing required"):
                    read_input_signature(handle, shape, "RBD")
            fixture(path, "RBD")
            with h5py.File(path, "r+") as handle:
                handle["X"][0, 0, 0, 0] = np.nan
            from scripts.network_training.neural.data import RbdRangeH5Dataset
            dataset = RbdRangeH5Dataset([path])
            try:
                with self.assertRaisesRegex(ValueError, "nonfinite features"):
                    dataset[0]
            finally:
                dataset.close()

    def test_prediction_exports_pdf_only(self):
        from scripts.network_training.neural.prediction import predict
        from scripts.network_training.common.prediction_utils import save_prediction_plot
        from scripts.network_training.neural.model import models_for
        for method in ("neural", "grnn"):
            module = importlib.import_module(f"scripts.network_training.{method}.cli")
            required = [] if method == "grnn" else ["--method", "RBD", "--model", models_for("RBD")[0]]
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                module.build_parser().parse_args(["predict", "--data", "test", *required, "--plot-path", "plot.png"])
        with self.assertRaisesRegex(ValueError, ".pdf"):
            save_prediction_plot([(0, 1, 0., 1., 1., 0.)], Path("plot.png"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.touch()
            signature = {"method": "RBD", "input_shape": [2, 1, 1]}
            checkpoint = {"checkpoint_version": 1, "method": "RBD", "model_name": "test_model", "dataset_name": "test",
                          "model_config": {}, "model_state": {}, "input_signature": signature,
                          "input_normalized_per_sample": False, "target_mean_km": 0., "target_std_km": 1.,
                          "loss_space": "km", "huber_beta": .5}
            dataset = TensorDataset(torch.ones(2, 2, 1, 1), torch.ones(2, 1))
            dataset.input_shape = (2, 1, 1)
            dataset.input_signature = signature
            dataset.source_segment_idx = [1, 2]
            dataset.window_center_s = [0., 100.]
            class ConstantModel(nn.Module):
                def forward(self, x):
                    return torch.ones(x.shape[0], 1, device=x.device)
            bindings = SimpleNamespace(method="RBD", dataset_class=lambda *a, **k: dataset,
                                       check_config_shape=lambda *a: None,
                                       model=SimpleNamespace(model_config_from_checkpoint=lambda c: ("test_model", None),
                                                             build_model=lambda *a: ConstantModel()))
            args = argparse.Namespace(batch_size=2, num_workers=0, print_first=0, data="test", output_dir=root,
                                      model="test_model", checkpoint=checkpoint_path, device="cpu", predictions_csv=None,
                                      plot_path=None, no_plot=False)
            with patch("scripts.network_training.neural.prediction.torch.load", return_value=checkpoint), patch("scripts.network_training.neural.prediction.resolve_h5_paths", return_value=[]), patch("scripts.network_training.neural.prediction.save_prediction_plot") as plot, patch("sys.stdout", io.StringIO()):
                predict(args, bindings)
                rows, path = plot.call_args.args
                self.assertEqual(path.suffix, ".pdf")
                self.assertIn("test_range_prediction_", path.name)
                self.assertEqual(len(rows), 2)
                self.assertEqual(plot.call_args.kwargs["title"], "RBD Source Range Prediction")
            # Render from the original rows, retaining both observations over the gap.
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            font_policy = matplotlib.rcParams["pdf.fonttype"]
            original_subplots = plt.subplots
            captured = []
            def capture_axes(*a, **k):
                figure, axes = original_subplots(*a, **k)
                captured.append(axes)
                return figure, axes
            pdf = root / "observations.pdf"
            with patch.object(plt, "subplots", side_effect=capture_axes):
                save_prediction_plot(rows, pdf)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            self.assertEqual(len(captured[0].lines), 0)
            self.assertEqual([len(collection.get_offsets()) for collection in captured[0].collections], [2, 2])
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], font_policy)
            self.assertFalse(list(root.rglob("*.png")))

    def test_prediction_contract_rejects_changed_axis(self):
        from scripts.data_processing.input_contract import require_matching_input_signature
        with self.assertRaisesRegex(ValueError, "frequency_hz"):
            require_matching_input_signature({"frequency_hz": [1, 2]}, {"frequency_hz": [1, 3]}, "Prediction checkpoint/data")

    def test_resume_zero_lr_updates_weights_and_retains_existing_best(self):
        @dataclass
        class Config:
            input_elements: int = 1
            input_freq_bins: int = 1

        model_api = SimpleNamespace(
            build_model_config=lambda *a, **k: Config(),
            build_model=lambda *a: nn.Sequential(nn.Flatten(), nn.Linear(2, 1)),
            model_config_from_checkpoint=lambda c: (c["model_name"], Config(**c["model_config"])),
            serialize_model_config=asdict, model_uses_dataset_input_norm=lambda name: False,
        )
        dataset = TensorDataset(torch.tensor([[[[1.]], [[2.]]], [[[2.]], [[1.]]]]), torch.tensor([[1.], [3.]]))
        signature = {"method": "RBD", "input_shape": [2, 1, 1]}
        def bundle(*args, **kwargs):
            return SimpleNamespace(input_shape=(2, 1, 1), input_signature=signature,
                                   train_dataset=dataset, val_dataset=dataset,
                                   train_idx=[0, 1], val_idx=[0, 1], train_labels=dataset.tensors[1],
                                   split_mode=kwargs["split_mode"])
        bindings = MethodBindings("RBD", model_api, SimpleNamespace(fixed_split_bundle=bundle), type(dataset), "input_elements")
        def controlled_epoch(*args, **kwargs):
            if len(args) == 8:  # training, including optimizer: real numerical update
                return run_epoch(*args, **kwargs)
            return {"loss": 1., "mae_km": 1., "rmse_km": 1.}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = argparse.Namespace(seed=2026, no_resume=True, resume_checkpoint=None, train_data=["train.h5"], val_data=["val.h5"], data="test",
                                      no_input_norm=False, model="test_model", val_fraction=.25, loss_space="km", batch_size=2, num_workers=0, device="cpu",
                                      base_channels=1, dropout=0., huber_beta=.5, lr=.1, weight_decay=0., epochs=1, output_dir=output)
            with patch("scripts.network_training.neural.training.resolve_h5_paths", return_value=[output / "dummy.h5"]), patch("scripts.network_training.neural.training.run_epoch", side_effect=controlled_epoch), patch("scripts.network_training.neural.training.time_suffix", side_effect=["0101_000000", "0101_000001"]), patch("sys.stdout", io.StringIO()) as stdout:
                train(args, bindings)
                first_path = output / "test_model/train_outputs/test_last_0101_000000.pt"
                first = torch.load(first_path, map_location="cpu")
                self.assertEqual(first["optimizer_state"]["param_groups"][0]["lr"], 0)
                args.no_resume = False; args.resume_checkpoint = first_path
                train(args, bindings)
                resumed = torch.load(output / "test_model/train_outputs/test_last_0101_000001.pt", map_location="cpu")
                self.assertTrue(any(not torch.equal(first["model_state"][k], resumed["model_state"][k]) for k in first["model_state"]))
                self.assertEqual(resumed["scheduler_plan"]["initial_lr"], .1)
                self.assertEqual(resumed["scheduler_plan"]["start_epoch"], 2)
                self.assertEqual(resumed["history"][-1]["lr"], .1)
                historical_best = output / "test_model/train_outputs/test_best_0101_000000.pt"
                self.assertEqual(Path(resumed["best_checkpoint_path"]), historical_best)
                self.assertTrue(historical_best.is_file())
                self.assertFalse((output / "test_model/train_outputs/test_best_0101_000001.pt").exists())
                self.assertIn(f"Best checkpoint: {historical_best}", stdout.getvalue())

    def test_blocked_split_indices_ordering_gap_and_rejections(self):
        # Out-of-order window centers: chronological order is [1, 5, 3, 2, 4, 0] -> [0, 5, 10, 20, 25, 30].
        centers = [30.0, 0.0, 20.0, 10.0, 25.0, 5.0]
        train_idx, val_idx = blocked_split_indices(centers, val_fraction=1 / 3, val_gap_s=0.0)
        self.assertEqual(train_idx, [1, 5, 3, 2])
        self.assertEqual(val_idx, [4, 0])

        # Stable tie-breaking and zero-gap retention even with equal timestamps across the boundary.
        self.assertEqual(
            blocked_split_indices([10.0, 20.0, 20.0, 30.0], val_fraction=0.5, val_gap_s=0.0),
            ([0, 1], [2, 3]),
        )

        # Exact 5.0 s separation between last train (t=20.0) and first val (t=25.0) is retained.
        train_exact, val_exact = blocked_split_indices(centers, val_fraction=1 / 3, val_gap_s=5.0)
        self.assertEqual(train_exact, [1, 5, 3, 2])
        self.assertEqual(val_exact, [4, 0])

        # Gap of 10.0 s drops sample 2 (t=20.0, gap 5.0 s) and keeps [0.0, 5.0, 10.0].
        train_gapped, val_gapped = blocked_split_indices(centers, val_fraction=1 / 3, val_gap_s=10.0)
        self.assertEqual(train_gapped, [1, 5, 3])
        self.assertEqual(val_gapped, [4, 0])

        # Excessive gap drops all candidate training samples and must fail cleanly.
        with self.assertRaisesRegex(ValueError, "empty train or validation partition"):
            blocked_split_indices(centers, val_fraction=1 / 3, val_gap_s=30.0)

        for bad_centers in ([], [5.0], [0.0, float("nan")], [0.0, float("inf")], [0.0, True]):
            with self.subTest(bad_centers=bad_centers), self.assertRaises(ValueError):
                blocked_split_indices(bad_centers, val_fraction=0.5, val_gap_s=0.0)
        for bad_fraction in (0.0, 1.0, -0.25, 1.5, float("nan"), float("inf"), True):
            with self.subTest(bad_fraction=bad_fraction), self.assertRaises(ValueError):
                blocked_split_indices([0.0, 10.0, 20.0, 30.0], val_fraction=bad_fraction, val_gap_s=0.0)
        for bad_gap in (-1.0, -1e-6, float("nan"), float("inf"), False, True):
            with self.subTest(bad_gap=bad_gap), self.assertRaises(ValueError):
                blocked_split_indices([0.0, 10.0, 20.0, 30.0], val_fraction=0.25, val_gap_s=bad_gap)

    def test_blocked_split_bundle_cli_and_resume_gap_contract(self):
        from scripts.network_training.neural.cli import build_parser
        from scripts.network_training.neural.data import RbdRangeH5Dataset, blocked_split_bundle
        from scripts.network_training.neural.model import models_for

        parser = build_parser()
        rbd_model = models_for("RBD")[0]
        default_args = parser.parse_args(["train", "--method", "RBD", "--model", rbd_model, "--data", "test"])
        self.assertEqual(default_args.val_split_mode, "random")
        self.assertEqual(default_args.val_gap_s, 0.0)

        blocked_args = parser.parse_args([
            "train", "--method", "RBD", "--model", rbd_model, "--data", "test",
            "--val-split-mode", "blocked", "--val-gap-s", "15.0",
        ])
        self.assertEqual(blocked_args.val_split_mode, "blocked")
        self.assertEqual(blocked_args.val_gap_s, 15.0)

        for bad_cli in (
            ["--val-split-mode", "unknown"],
            ["--val-gap-s", "-1"],
            ["--val-gap-s", "nan"],
            ["--val-gap-s", "inf"],
        ):
            with self.subTest(bad_cli=bad_cli), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["train", "--method", "RBD", "--model", rbd_model, "--data", "test", *bad_cli])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "blocked.h5"
            fixture(h5_path, "RBD")
            with h5py.File(h5_path, "r+") as handle:
                del handle["X"], handle["y_range_km"], handle["split/source_segment_idx"], handle["time/window_center_s"]
                values = np.ones((6, 3, 5, 2), dtype=np.float32)
                x = handle.create_dataset("X", data=values)
                x.attrs["axis_order"] = "sample,element,frequency,real_imag"
                x.attrs["storage_axis_order"] = "sample,element,frequency,real_imag"
                handle["y_range_km"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
                handle["split/source_segment_idx"] = np.arange(6) + 1
                handle["time/window_center_s"] = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)

            bundle = blocked_split_bundle(
                RbdRangeH5Dataset,
                [h5_path],
                normalize_input=False,
                val_fraction=1 / 3,
                val_gap_s=15.0,
                split_mode="dataset_train_blocked_split",
            )
            try:
                # val_idx=[4, 5] starts at 40.0 s; sample 3 (30.0 s, gap 10.0 s < 15.0 s) is dropped.
                self.assertEqual(bundle.train_idx, [0, 1, 2])
                self.assertEqual(bundle.val_idx, [4, 5])
                self.assertEqual(len(bundle.train_dataset), 3)
                self.assertEqual(len(bundle.val_dataset), 2)
                self.assertTrue(torch.equal(bundle.train_labels, torch.tensor([[1.0], [2.0], [3.0]])))
                self.assertEqual(bundle.split_mode, "dataset_train_blocked_split")
            finally:
                bundle.dataset.close()

            @dataclass
            class Config:
                input_elements: int = 3
                input_freq_bins: int = 5

            model_api = SimpleNamespace(
                build_model_config=lambda *a, **k: Config(),
                build_model=lambda *a: nn.Sequential(nn.Flatten(), nn.Linear(30, 1)),
                model_config_from_checkpoint=lambda c: (c["model_name"], Config(**c["model_config"])),
                serialize_model_config=asdict,
                model_uses_dataset_input_norm=lambda name: False,
            )
            opened_datasets = []

            def tracked_blocked_bundle(*b_args, **b_kwargs):
                created = blocked_split_bundle(*b_args, **b_kwargs)
                opened_datasets.append(created.dataset)
                return created

            data_dir = root / "test"
            data_dir.mkdir()
            data_api = SimpleNamespace(blocked_split_bundle=tracked_blocked_bundle)
            bindings = MethodBindings("RBD", model_api, data_api, RbdRangeH5Dataset, "input_elements")
            output = root / "out"
            args = argparse.Namespace(
                seed=2026, no_resume=True, resume_checkpoint=None, train_data=[str(h5_path)], val_data=None,
                data=str(data_dir), no_input_norm=True, model="complex_cnn_range", val_fraction=1 / 3,
                val_split_mode="blocked", val_gap_s=12.0, loss_space="km", batch_size=2, num_workers=0,
                device="cpu", base_channels=1, dropout=0.0, huber_beta=0.5, lr=0.01, weight_decay=0.0,
                epochs=1, output_dir=output,
            )
            try:
                with patch("scripts.network_training.neural.training.time_suffix", return_value="0101_000010"), patch("sys.stdout", io.StringIO()):
                    train(args, bindings)
                saved_last = output / "complex_cnn_range/train_outputs/test_last_0101_000010.pt"
                ckpt = torch.load(saved_last, map_location="cpu")
                self.assertEqual(ckpt["split_mode"], "explicit_train_data_blocked_split")
                self.assertEqual(ckpt["val_split_mode"], "blocked")
                self.assertEqual(ckpt["val_gap_s"], 12.0)
                self.assertEqual(ckpt["train_indices"], [0, 1, 2])
                self.assertEqual(ckpt["val_indices"], [4, 5])

                # Changing val_gap_s from 12.0 to 15.0 produces identical indices [0, 1, 2] / [4, 5],
                # yet resume must still reject the mismatched val_gap_s setting.
                args.no_resume = False
                args.resume_checkpoint = saved_last
                args.val_gap_s = 15.0
                with self.assertRaisesRegex(ValueError, "val_gap_s"):
                    train(args, bindings)
            finally:
                for ds in opened_datasets:
                    ds.close()

    def test_lazy_model_registry_defers_resnet_import(self):
        project_root = Path(__file__).resolve().parents[4]
        code = """import sys
from scripts.network_training.neural.model import MODEL_REGISTRY, build_model_config, models_for
from scripts.network_training.neural.cli import build_parser

expected = [
    "complex_cnn_range",
    "resnet18_range",
    "resnet50_range",
    "scm_complex_cnn_range",
    "scm_resnet18_range",
    "scm_resnet50_range",
]
assert sorted(MODEL_REGISTRY) == expected
assert len(MODEL_REGISTRY) == 6
assert "resnet18_range" in MODEL_REGISTRY
assert "unknown_model" not in MODEL_REGISTRY
assert list(MODEL_REGISTRY.keys()) == list(MODEL_REGISTRY)
assert len(models_for("RBD")) == 3 and len(models_for("SCM")) == 3
build_model_config("resnet18_range", (2, 21, 23))
build_parser()
assert "scripts.network_training.neural.models.resnet" not in sys.modules
assert "scripts.network_training.neural.models.complex_cnn" not in sys.modules

cnn_cfg, cnn_cls = MODEL_REGISTRY["scm_complex_cnn_range"]
assert cnn_cls.__name__ == "ComplexRangeCNN"
assert "scripts.network_training.neural.models.complex_cnn" in sys.modules
assert "scripts.network_training.neural.models.resnet" not in sys.modules

rn_cfg, rn_cls = MODEL_REGISTRY["resnet18_range"]
assert rn_cls.__name__ == "ResNet18Range"
assert "scripts.network_training.neural.models.resnet" in sys.modules
"""
        proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=project_root, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
