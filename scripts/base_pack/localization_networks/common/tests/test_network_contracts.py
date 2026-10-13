"""Critical rejection and continuation cases, executed on haiqin only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.base_pack.localization_networks.common.bindings import MethodBindings
from scripts.base_pack.localization_networks.common.checkpoints import validate_checkpoint
from scripts.base_pack.localization_networks.common.h5_utils import split_indices
from scripts.base_pack.localization_networks.common.input_contract import read_input_signature
from scripts.base_pack.localization_networks.common.training import train
from scripts.base_pack.localization_networks.common.training_utils import run_epoch


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
        for method in ("rbd", "elm", "scm"):
            module = importlib.import_module(f"scripts.base_pack.localization_networks.{method}.network.cli")
            model = module.MODEL_CHOICES[0]
            parser = module.build_parser()
            for extra in (["--no-resume", "--resume-checkpoint", "x.pt"], ["--epochs", "0"],
                          ["--val-fraction", "0"], ["--val-fraction", "nan"], ["--lr", "nan"],
                          ["--dropout", "1"], ["--num-workers", "-1"], ["--seed", "-1"],
                          ["--seed", str(2**32)], ["--ep", "1"]):
                with self.subTest(method=method, extra=extra), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["train", "--model", model, "--data", "test", *extra])

    def test_seed_range_matches_public_workflow(self):
        from scripts.base_pack.localization_networks.common.training_utils import seed_everything
        for seed in (-1, 2**32, False, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                seed_everything(seed)
        from scripts.base_pack.localization_networks.common.cli import seed_int
        self.assertEqual(seed_int("0"), 0)
        self.assertEqual(seed_int(str(2**32 - 1)), 2**32 - 1)

    def test_explicit_missing_prediction_checkpoint_never_uses_timestamp_match(self):
        from scripts.base_pack.localization_networks.common.prediction import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "requested.pt"
            # A genuine matching candidate exists; it must not satisfy an explicit path.
            (root / "requested_0101_000001.pt").write_bytes(b"different checkpoint")
            args = argparse.Namespace(batch_size=1, num_workers=0, print_first=0, data="test",
                                      output_dir=root, model="test_model", checkpoint=missing, device="cpu")
            with patch("scripts.base_pack.localization_networks.common.prediction.latest_time_suffixed_path") as latest, patch("scripts.base_pack.localization_networks.common.prediction.torch.load") as load:
                with self.assertRaisesRegex(FileNotFoundError, "Explicit prediction checkpoint"):
                    predict(args, SimpleNamespace(method="RBD"))
                latest.assert_not_called()
                load.assert_not_called()

    def test_missing_checkpoint_contract_and_wrong_model_fail(self):
        with self.assertRaisesRegex(ValueError, "missing contract fields"):
            validate_checkpoint({"model_name": "complex_cnn_range"}, method="RBD", model_name="complex_cnn_range", dataset_name="test")
        checkpoint = {"checkpoint_version": 1, "method": "RBD", "model_name": "real_cnn_range", "dataset_name": "test",
                      "model_config": {}, "model_state": {}, "input_signature": {"method": "RBD"},
                      "input_normalized_per_sample": False, "target_mean_km": 0, "target_std_km": 1,
                      "loss_space": "km", "huber_beta": .5}
        with self.assertRaisesRegex(ValueError, "model_name"):
            validate_checkpoint(checkpoint, method="RBD", model_name="complex_cnn_range", dataset_name="test")

    def test_same_shape_different_frequency_rejected_across_files_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method, cls_name in (("RBD", "RbdRangeH5Dataset"), ("ELM", "ElmRangeH5Dataset"), ("SCM", "ScmRangeH5Dataset")):
                module = importlib.import_module(f"scripts.base_pack.localization_networks.{method.lower()}.network.data")
                a, b = Path(tmp) / "a.h5", Path(tmp) / "b.h5"
                fixture(a, method); fixture(b, method, frequency_offset=1)
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "input signature mismatch"):
                    getattr(module, cls_name)([a, b])
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "input signature mismatch"):
                    module.fixed_split_bundle([a], [b], normalize_input=False, split_mode="fixed")

    def test_pair_semantics_missing_metadata_and_nonfinite_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.h5"
            for method in ("ELM", "SCM"):
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
            from scripts.base_pack.localization_networks.rbd.network.data import RbdRangeH5Dataset
            dataset = RbdRangeH5Dataset([path])
            try:
                with self.assertRaisesRegex(ValueError, "nonfinite features"):
                    dataset[0]
            finally:
                dataset.close()

    def test_prediction_exports_pdf_only(self):
        from scripts.base_pack.localization_networks.common.prediction import predict
        from scripts.base_pack.localization_networks.common.prediction_utils import save_prediction_plot
        for method in ("rbd", "elm", "scm", "grnn"):
            module_name = f"scripts.base_pack.localization_networks.{method}." + ("cli" if method == "grnn" else "network.cli")
            module = importlib.import_module(module_name)
            required = [] if method == "grnn" else ["--model", module.MODEL_CHOICES[0]]
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
            with patch("scripts.base_pack.localization_networks.common.prediction.torch.load", return_value=checkpoint), patch("scripts.base_pack.localization_networks.common.prediction.resolve_h5_paths", return_value=[]), patch("scripts.base_pack.localization_networks.common.prediction.save_prediction_plot") as plot, patch("sys.stdout", io.StringIO()):
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
        from scripts.base_pack.localization_networks.common.input_contract import require_matching_input_signature
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
        bindings = MethodBindings("RBD", model_api, SimpleNamespace(fixed_split_bundle=bundle), SimpleNamespace(), type(dataset), "input_elements")
        def controlled_epoch(*args, **kwargs):
            if len(args) == 8:  # training, including optimizer: real numerical update
                return run_epoch(*args, **kwargs)
            return {"loss": 1., "mae_km": 1., "rmse_km": 1.}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = argparse.Namespace(seed=2026, no_resume=True, resume_checkpoint=None, train_data=["train.h5"], val_data=["val.h5"], data="test",
                                      no_input_norm=False, model="test_model", val_fraction=.25, loss_space="km", batch_size=2, num_workers=0, device="cpu",
                                      base_channels=1, dropout=0., huber_beta=.5, lr=.1, weight_decay=0., epochs=1, output_dir=output)
            with patch("scripts.base_pack.localization_networks.common.training.resolve_h5_paths", return_value=[output / "dummy.h5"]), patch("scripts.base_pack.localization_networks.common.training.run_epoch", side_effect=controlled_epoch), patch("scripts.base_pack.localization_networks.common.training.time_suffix", side_effect=["0101_000000", "0101_000001"]), patch("sys.stdout", io.StringIO()) as stdout:
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


if __name__ == "__main__":
    unittest.main()
