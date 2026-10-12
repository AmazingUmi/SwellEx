"""Method-aware range-overlap generation, reference ownership, and delivery contracts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from scipy.io import savemat

from scripts.localization_networks.grnn.cli import load_artifact, resolve_reference_path
from scripts.localization_networks.grnn.bindings import METHODS, artifact_type, model_name
from scripts.localization_networks.grnn.data import load_grnn_features
from scripts.localization_networks.grnn.model import GrnnRegressor
from scripts.localization_networks.workflow.local_grnn import run
from scripts.localization_networks.workflow.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "scripts/localization_networks/workflow"


def write_numeric_inputs(root: Path) -> None:
    """Write small numeric inputs through the same formats as the S5 recording."""
    origin = root / "Origindata"
    channels = origin / "events/S5/vla_matfiles"
    positions = origin / "positions/positions_vla.txt"
    range_file = origin / "events/range/RangeEventS5/SproulToVLA.S5.txt"
    ctd = origin / "environments/ctds/i9605.prn"
    for path in (channels, positions.parent, range_file.parent, ctd.parent):
        path.mkdir(parents=True, exist_ok=True)

    fs = 100
    times = np.arange(40 * fs, dtype=np.float64) / fs
    ranges = np.where(times <= 19.5, 1 + (19.5 - times) * 0.1,
                      1 + (times - 19.5) * 0.12)
    generator = np.random.default_rng(2026)
    for element in range(1, 4):
        signal = (np.cos(2 * np.pi * 49 * times + 0.2 * element * ranges)
                  + 0.15 * np.cos(2 * np.pi * 50 * times + 0.09 * element * times)
                  + 0.005 * generator.normal(size=len(times)))
        savemat(channels / f"S5_VLA_NO_{element}.mat", {"x": signal.reshape(-1, 1)})
    # load_inputs reverses the position-file depth axis before feature generation.
    np.savetxt(positions, [[1, 30], [2, 20], [3, 10]])
    centers = np.arange(40, dtype=np.float64) + 0.5
    range_times = np.concatenate(([0], centers, [40]))
    range_values = np.where(range_times <= 19.5, 1 + (19.5 - range_times) * 0.1,
                            1 + (range_times - 19.5) * 0.12)
    np.savetxt(range_file, np.column_stack((np.zeros_like(range_times),
                                          np.zeros_like(range_times), range_times / 60,
                                          range_values)),
               header="latitude longitude time_min range_km", comments="")
    np.savetxt(ctd, [[0, 0, 0, 1500, 0], [40, 0, 0, 1500, 0]])


def write_small_config(root: Path, side: str, *, split: str = "range_overlap") -> Path:
    """Keep the complete existing workflow contract while reducing fixture size."""
    config = load_config(CONFIG_DIR / f"haiqin_range_overlap_train_{side}.toml")
    config["validation"].update(device="cpu", threads=1)
    config["dataset"].update(fs=100, segment_end_s=39, half_duration_s=14,
                             gap_s=1, frequency_selection_modes=["deep"],
                             split_strategy=split)
    config["grnn"].update(cv_folds=3, sigma_candidates=[0.01, 0.1, 1.0], batch_size=4)
    lines = [f"schema_version = {config['schema_version']}"]
    for table, values in config.items():
        if table == "schema_version":
            continue
        lines.extend(("", f"[{table}]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in values.items())
    path = root / f"{split}_{side}.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    load_config(path)
    return path


class RangeOverlapLocalContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_numeric_inputs(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_delivered_run(self, report: dict, side: str) -> None:
        # The asymmetric range curve has 14 candidates on each side. Filtering
        # to the shared [1.24, 2.5] km interval retains 13 before and 11 after.
        before = list(range(5, 18))
        after = list(range(22, 33))
        train_indices, test_indices = (before, after) if side == "before" else (after, before)
        method = report["method"]
        self.assertIn(method, METHODS)
        self.assertEqual(report["model"], model_name(method))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["execution"]["device"], "cpu")
        run_dir = self.root / "output/data" / report["run_id"]
        self.assertEqual(json.loads((run_dir / "report.json").read_text()), report)
        dataset = report["dataset"]
        self.assertEqual(dataset["train_count"], len(train_indices))
        self.assertEqual(dataset["test_count"], len(test_indices))
        split = dataset["split"]
        self.assertEqual(split["split_strategy"], "range_overlap")
        self.assertEqual(split["range_overlap_train_side"], side)
        self.assertEqual(split["range_overlap_candidate_num_segments_before"], 14)
        self.assertEqual(split["range_overlap_candidate_num_segments_after"], 14)
        self.assertEqual(split["range_overlap_num_segments_before"], 13)
        self.assertEqual(split["range_overlap_num_segments_after"], 11)
        self.assertAlmostEqual(split["range_overlap_min_km"], 1.24)
        self.assertAlmostEqual(split["range_overlap_max_km"], 2.5)
        identity = report["split_identity"]
        self.assertEqual(identity["train_source_segment_idx"], train_indices)
        self.assertEqual(identity["test_source_segment_idx"], test_indices)
        self.assertTrue(set(train_indices).isdisjoint(test_indices))
        self.assertEqual(identity["train_window_center_s"], [i - 0.5 for i in train_indices])
        self.assertEqual(identity["test_window_center_s"], [i - 0.5 for i in test_indices])
        train_path, test_path = Path(dataset["train_file"]), Path(dataset["test_file"])
        for path, expected in ((train_path, train_indices), (test_path, test_indices)):
            with h5py.File(path, "r") as handle:
                height = 6 if method == "SCM" else 3
                self.assertEqual(handle["X"].shape, (len(expected), height, 2, 2))
                self.assertEqual(handle["split/source_segment_idx"][:].reshape(-1).tolist(), expected)
                labels = handle["y_range_km"][:].reshape(-1)
                self.assertTrue(np.all(labels >= 1.24 - 1e-6))
                self.assertTrue(np.all(labels <= 2.5 + 1e-6))
        datasets_root = Path(dataset["directory"]).parent
        self.assertEqual(len(list(datasets_root.glob("*/*_train.h5"))), 1)
        self.assertEqual(len(list(datasets_root.glob("*/*_test.h5"))), 1)
        self.assertTrue(all(f"{method.lower()}_" in p.name for p in datasets_root.iterdir()))

        reference = report["reference"]
        artifact = load_artifact(Path(reference["path"]), torch.device("cpu"), method=method)
        train = load_grnn_features([train_path], method=method)
        test = load_grnn_features([test_path], method=method)
        self.assertEqual(artifact["artifact_type"], artifact_type(method))
        self.assertEqual(artifact["method"], model_name(method))
        self.assertEqual(artifact["input_signature"]["method"], method)
        self.assertEqual(reference["reference_sample_count"], len(train_indices))
        self.assertEqual(reference["sigma_source"], "cross_validation")
        self.assertIn(reference["sigma"], [0.01, 0.1, 1.0])
        self.assertEqual(artifact["reference_sample_count"], len(train_indices))
        self.assertEqual(artifact["source_h5_paths"], [str(train_path)])
        self.assertEqual(artifact["split_strategy"], "range_overlap")
        self.assertEqual(artifact["train_side"], side)
        self.assertEqual(artifact["reference_source_segment_idx"], train_indices)
        self.assertEqual(artifact["reference_window_center_s"], [i - 0.5 for i in train_indices])
        torch.testing.assert_close(artifact["y_reference_km"], train.y_range_km)
        self.assertFalse(artifact["standardize_input"])
        torch.testing.assert_close(artifact["x_reference"], train.x)
        self.assertEqual(artifact["input_signature"], test.input_signature)

        predictions = report["predictions"]
        self.assertEqual(predictions["samples"], len(test_indices))
        with Path(predictions["path"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["source_segment_idx"]) for row in rows], test_indices)
        self.assertEqual(len(rows), len(test_indices))
        predicted = GrnnRegressor.from_artifact(artifact).predict(test.x, batch_size=4).view(-1).numpy()
        np.testing.assert_allclose([float(row["pred_range_km"]) for row in rows], predicted, atol=1e-6)
        np.testing.assert_allclose([float(row["true_range_km"]) for row in rows],
                                   test.y_range_km.view(-1).numpy(), atol=1e-6)
        self.assertTrue(all(np.isfinite(predictions[key]) for key in ("mae_km", "rmse_km", "mape_pct")))
        search_paths = [Path(item["path"]) for item in report["artifacts"]
                        if "sigma_search" in Path(item["path"]).name and Path(item["path"]).suffix == ".json"]
        self.assertEqual(len(search_paths), 1)
        self.assertEqual([entry["sigma"] for entry in json.loads(search_paths[0].read_text())], [0.01, 0.1, 1.0])
        for entry in report["artifacts"]:
            path = Path(entry["path"])
            self.assertEqual(entry["size_bytes"], path.stat().st_size)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(report["checks"])
        self.assertTrue(all(check["status"] == "PASS" for check in report["checks"]))

    def test_before_and_after_keep_complete_independent_references(self):
        for method in METHODS:
            for side in ("before", "after"):
                with self.subTest(method=method, side=side), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    write_numeric_inputs(root)
                    config = write_small_config(root, side)
                    with patch("scripts.localization_networks.grnn.cli.resolve_reference_path",
                               wraps=resolve_reference_path) as resolve:
                        report = run(config, method=method, root=root)
                    # Prediction must bind the artifact produced by this build.
                    self.assertEqual(resolve.call_count, 1)
                    self.assertEqual(Path(resolve.call_args.args[0].reference), Path(report["reference"]["path"]))
                    original_root = self.root
                    try:
                        self.root = root
                        self.assert_delivered_run(report, side)
                    finally:
                        self.root = original_root

    def test_reused_dataset_rejects_opposite_training_side(self):
        before = write_small_config(self.root, "before")
        report = run(before, method="SCM", root=self.root)
        after = write_small_config(self.root, "after")
        with self.assertRaisesRegex(ValueError, "train_side|configuration|config|identity|mismatch"):
            run(after, method="SCM", data=Path(report["dataset"]["directory"]), root=self.root)

    def test_same_side_reuse_preserves_original_delivery_and_hdf5(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                original_root = self.root
                self.root = Path(temporary)
                try:
                    write_numeric_inputs(self.root)
                    config = write_small_config(self.root, "before")
                    original = run(config, method=method, root=self.root)
                    paths = [Path(item["path"]) for item in [*original["artifacts"], *original["input_files"]]]
                    paths.append(self.root / "output/data" / original["run_id"] / "report.json")
                    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                    reused = run(config, method=method, data=Path(original["dataset"]["directory"]), root=self.root)
                    self.assertNotEqual(reused["run_id"], original["run_id"])
                    self.assertNotEqual(reused["reference"]["path"], original["reference"]["path"])
                    self.assertEqual(reused["dataset"]["directory"], original["dataset"]["directory"])
                    self.assertEqual(reused["input_files"], original["input_files"])
                    self.assertNotIn(f"generate_{method.lower()}", [check["phase"] for check in reused["checks"]])
                    self.assert_delivered_run(reused, "before")
                    for path, digest in hashes.items():
                        with self.subTest(path=path):
                            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                finally:
                    self.root = original_root

    def test_reuse_rejects_another_method_and_changed_rbd_estimator(self):
        config = write_small_config(self.root, "before")
        original = run(config, method="RBD", root=self.root)
        data = Path(original["dataset"]["directory"])
        with self.assertRaisesRegex(ValueError, "feature method"):
            run(config, method="ELM", data=data, root=self.root)
        changed = self.root / "changed_estimator.toml"
        changed.write_text(config.read_text().replace('beam_selection = "best"', 'beam_selection = "multipath"'))
        with self.assertRaisesRegex(ValueError, "RBD.*configuration|rbd.*configuration|estimator"):
            run(changed, method="RBD", data=data, root=self.root)

    def test_reused_dataset_rejects_different_raw_range_track(self):
        config = write_small_config(self.root, "before")
        report = run(config, method="SCM", root=self.root)
        mixed = self.root / "mixed"
        shutil.copytree(report["dataset"]["directory"], mixed)
        test_path = mixed / Path(report["dataset"]["test_file"]).name
        with h5py.File(test_path, "r+") as handle:
            # Labels remain internally consistent with their own raw track;
            # rejecting this mix requires comparing train and test provenance.
            handle["range/range_km_raw"][:] += 10
            handle["y_range_km"][:] += 10
        with self.assertRaisesRegex(ValueError, "range track"):
            run(config, method="SCM", data=mixed, root=self.root)

    def test_full_interleaved_is_rejected_before_loading_inputs(self):
        config = write_small_config(self.root, "before", split="full_interleaved")
        with self.assertRaisesRegex(ValueError, "range_overlap"):
            run(config, method="SCM", root=self.root / "no_raw_inputs")
        self.assertFalse((self.root / "no_raw_inputs/outputs/Datasets").exists())


if __name__ == "__main__":
    unittest.main()
