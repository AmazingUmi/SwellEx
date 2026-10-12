"""Three-method GRNN feature, artifact, and command contracts; run on haiqin."""
from __future__ import annotations

import copy
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.localization_networks.grnn.bindings import METHODS, artifact_type, model_name, validate_method
from scripts.localization_networks.grnn.cli import (
    build_parser, load_artifact, predict_command, validate_arguments,
)
from scripts.localization_networks.grnn.data import load_grnn_features
from scripts.localization_networks.grnn.model import GrnnRegressor
from scripts.localization_networks.grnn.paths import default_output_dir, reference_output_dir, test_output_dir


def write_features(path: Path, method: str, *, reversed_storage: bool = False, frequencies=(10.0, 20.0)):
    elements = 3
    pairs = [(i, j) for j in range(1, elements + 1)
             for i in range(1, j + (method == "SCM"))]
    height = elements if method == "RBD" else len(pairs)
    x = np.arange(4 * height * 2 * 2, dtype=np.float32).reshape(4, height, 2, 2) / 100
    axis = "element" if method == "RBD" else "pair"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("X", data=x.transpose(3, 2, 1, 0) if reversed_storage else x)
        dataset.attrs["axis_order"] = f"sample,{axis},frequency,real_imag"
        dataset.attrs["storage_axis_order"] = (
            f"real_imag,frequency,{axis},sample" if reversed_storage
            else f"sample,{axis},frequency,real_imag"
        )
        values = {
            "y_range_km": [1.0, 2.0, 3.0, 4.0], "valid_sample": [1, 1, 0, 1],
            "frequency/freq_hz": frequencies, "array/element_idx": [1, 2, 3],
            "array/depth_m": [10.0, 20.0, 30.0],
            "split/source_segment_idx": [11, 12, 13, 14],
            "time/window_center_s": [0.5, 1.5, 2.5, 3.5],
        }
        if method != "RBD":
            values["pair/numerator_element_idx"] = [i for i, _ in pairs]
            values["pair/denominator_element_idx"] = [j for _, j in pairs]
        for key, value in values.items():
            handle.create_dataset(key, data=np.asarray(value).reshape(-1, 1))
    return x


def make_artifact(bundle, method: str):
    model = GrnnRegressor(spread=0.1)
    model.fit_reference(bundle.x, bundle.y_range_km)
    return dict(model.to_artifact(), artifact_type=artifact_type(method), method=model_name(method),
                input_shape=bundle.input_shape, input_signature=bundle.input_signature,
                feature_dim=bundle.x.size(1), reference_sample_count=len(bundle.x),
                per_sample_channel_norm=False)


class MethodContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_all_native_and_reversed_layouts_preserve_raw_complex_values(self):
        for method in METHODS:
            with self.subTest(method=method):
                native, reverse = (self.root / f"{method}_{name}.h5" for name in ("native", "reverse"))
                raw = write_features(native, method)
                write_features(reverse, method, reversed_storage=True)
                first = load_grnn_features([native], method=method)
                second = load_grnn_features([reverse], method=method)
                expected = raw[[0, 1, 3]].transpose(0, 3, 1, 2).reshape(3, -1)
                np.testing.assert_array_equal(first.x.numpy(), expected)
                torch.testing.assert_close(first.x, second.x, rtol=0, atol=0)
                self.assertEqual(first.input_signature, second.input_signature)
                self.assertEqual(first.input_signature["method"], method)
                self.assertEqual(first.source_segment_idx, [11, 12, 14])
                self.assertEqual(first.window_center_s, [0.5, 1.5, 3.5])
                limited = load_grnn_features([native], method=method, max_samples=2)
                torch.testing.assert_close(limited.x, first.x[:2], rtol=0, atol=0)

    def test_scm_preprocessing_and_kernel_match_previous_calculation(self):
        for reverse in (False, True):
            path = self.root / f"SCM_{reverse}.h5"
            raw = write_features(path, "SCM", reversed_storage=reverse)
            for normalized in (False, True):
                with self.subTest(normalized=normalized, reversed_storage=reverse):
                    previous = []
                    for sample in (0, 1, 3):
                        # Previous loader read one stored sample and transposed
                        # its native/reversed axes before population mean/std.
                        stored = raw[sample].transpose(2, 1, 0).copy() if reverse else raw[sample]
                        value = stored.transpose(0, 2, 1) if reverse else stored.transpose(2, 0, 1)
                        if normalized:
                            mean = value.mean(axis=(1, 2), keepdims=True)
                            std = value.std(axis=(1, 2), keepdims=True)
                            value = (value - mean) / np.maximum(std, 1e-6)
                        previous.append(value.reshape(-1))
                    expected = np.stack(previous)
                    bundle = load_grnn_features([path], method="SCM", per_sample_channel_norm=normalized)
                    np.testing.assert_array_equal(bundle.x.numpy(), expected)
                    model = GrnnRegressor(spread=0.3)
                    model.fit_reference(bundle.x, bundle.y_range_km)
                    query = bundle.x[:2] + 0.04
                    distances = torch.cdist(query, torch.from_numpy(expected)).square()
                    distances -= distances.min(dim=1, keepdim=True).values
                    weights = torch.exp(-distances / (2 * 0.3**2))
                    prediction = weights @ bundle.y_range_km / weights.sum(dim=1, keepdim=True)
                    torch.testing.assert_close(model.predict(query, batch_size=2), prediction, rtol=0, atol=0)

    def test_pair_relations_and_order_are_method_specific(self):
        for method in ("ELM", "SCM"):
            with self.subTest(method=method):
                path = self.root / f"{method}.h5"
                write_features(path, method)
                bundle = load_grnn_features([path], method=method)
                expected = [[i, j] for j in range(1, 4) for i in range(1, j + (method == "SCM"))]
                self.assertEqual(bundle.input_signature["pair_indices"], expected)
                with h5py.File(path, "r+") as handle:
                    if method == "ELM":
                        handle["pair/denominator_element_idx"][0, 0] = 1
                    else:
                        handle["pair/numerator_element_idx"][1, 0] = 2
                with self.assertRaisesRegex(ValueError, "pairs must"):
                    load_grnn_features([path], method=method)

    def test_same_shape_elm_without_axis_declarations_cannot_be_loaded_as_rbd(self):
        # E=3 gives both RBD element count and ELM pair count equal to three.
        # Complete pair metadata identifies the method even without axis attrs.
        for reverse in (False, True):
            with self.subTest(reversed_storage=reverse):
                path = self.root / f"ambiguous_{reverse}.h5"
                write_features(path, "ELM", reversed_storage=reverse)
                with h5py.File(path, "r+") as handle:
                    del handle["X"].attrs["axis_order"]
                    del handle["X"].attrs["storage_axis_order"]
                actual = load_grnn_features([path], method="ELM")
                self.assertEqual(actual.input_shape, (2, 3, 2))
                with self.assertRaisesRegex(ValueError, "RBD.*pair metadata"):
                    load_grnn_features([path], method="RBD")
                # Any remaining pair field rejects; do not infer RBD when one
                # field of an invalid paired source has been removed.
                with h5py.File(path, "r+") as handle:
                    del handle["pair/numerator_element_idx"]
                with self.assertRaisesRegex(ValueError, "RBD.*pair metadata"):
                    load_grnn_features([path], method="RBD")

    def test_artifacts_reject_other_methods_even_when_shapes_match(self):
        for method in METHODS:
            with self.subTest(method=method):
                source = self.root / f"{method}.h5"
                write_features(source, method)
                bundle = load_grnn_features([source], method=method)
                artifact = make_artifact(bundle, method)
                path = self.root / f"{method}.pt"
                torch.save(artifact, path)
                self.assertEqual(load_artifact(path, torch.device("cpu"), method=method)["method"], model_name(method))
                for other in METHODS:
                    if other != method:
                        with self.assertRaisesRegex(ValueError, "method mismatch"):
                            load_artifact(path, torch.device("cpu"), method=other)
                for key, replacement in (("method", "ELM" if method == "RBD" else "RBD"),
                                         ("frequency_hz", [20.0, 10.0]),
                                         ("depth_m", [30.0, 20.0, 10.0])):
                    bad = copy.deepcopy(artifact)
                    bad["input_signature"][key] = replacement
                    torch.save(bad, path)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        load_artifact(path, torch.device("cpu"), method=method)

    def test_prediction_rejects_changed_physical_axis_before_kernel_execution(self):
        for method in METHODS:
            with self.subTest(method=method):
                train = self.root / f"{method}_train.h5"
                changed = self.root / f"{method}_changed.h5"
                write_features(train, method)
                write_features(changed, method, frequencies=(11.0, 20.0))
                bundle = load_grnn_features([train], method=method)
                reference = self.root / f"{method}_reference.pt"
                torch.save(make_artifact(bundle, method), reference)
                parser = build_parser()
                args = parser.parse_args(["predict", "--method", method, "--data", "fixture",
                                          "--reference", str(reference), "--input-data", str(changed),
                                          "--output-dir", str(self.root / method), "--device", "cpu", "--no-plot"])
                validate_arguments(args, parser)
                with self.assertRaisesRegex(ValueError, "input signature mismatch"):
                    predict_command(args)
                self.assertFalse(test_output_dir(self.root / method, method=method).exists())

    def test_cli_build_predict_and_output_paths_are_method_specific(self):
        for method in METHODS:
            with self.subTest(method=method):
                source = self.root / f"{method}_train.h5"
                write_features(source, method)
                parser = build_parser()
                args = parser.parse_args(["build", "--method", method, "--data", "fixture",
                                          "--train-data", str(source), "--device", "cpu"])
                validate_arguments(args, parser)
                self.assertEqual(args.output_dir, default_output_dir(method))
                args.output_dir = self.root / method
                with redirect_stdout(io.StringIO()):
                    args.func(args)
                references = list(reference_output_dir(args.output_dir, method=method).glob("*_reference_*.pt"))
                self.assertEqual(len(references), 1)
                loaded = load_artifact(references[0], torch.device("cpu"), method=method)
                self.assertEqual(loaded["artifact_type"], artifact_type(method))
                self.assertFalse(loaded["per_sample_channel_norm"])
                self.assertFalse(loaded["standardize_input"])
                prediction = parser.parse_args(["predict", "--method", method, "--data", "fixture",
                                                "--input-data", str(source), "--device", "cpu",
                                                "--output-dir", str(args.output_dir), "--no-plot", "--print-first", "0"])
                validate_arguments(prediction, parser)
                with redirect_stdout(io.StringIO()):
                    prediction.func(prediction)
                paths = list(test_output_dir(args.output_dir, method=method).glob("*_predictions_*.csv"))
                self.assertEqual(len(paths), 1)
                with paths[0].open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
                self.assertEqual(reference_output_dir(args.output_dir / model_name(method), method=method),
                                 reference_output_dir(args.output_dir, method=method))

    def test_method_is_required_and_obsolete_names_are_absent(self):
        for method in ("rbd", "ELM ", "", None, True):
            with self.subTest(method=method), self.assertRaises(ValueError):
                validate_method(method)
        for command in ("build", "predict"):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                build_parser().parse_args([command, "--data", "fixture"])
        from scripts.localization_networks.grnn import data, model, paths
        self.assertFalse(hasattr(data, "load_scm_grnn_features"))
        self.assertFalse(hasattr(model, "ScmGrnnRegressor"))
        self.assertFalse(hasattr(paths, "DEFAULT_OUTPUT_DIR"))


if __name__ == "__main__":
    unittest.main()
