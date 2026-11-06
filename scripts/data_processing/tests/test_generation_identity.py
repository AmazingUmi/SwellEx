"""Generation identity and atomic output ownership; execute on haiqin."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from scipy.io import savemat

from scripts.data_processing.dataset_validation import run_datasets_command
from scripts.data_processing.generation_config import normalize_config
from scripts.data_processing.generation_core import (
    create_h5, dataset_names, generate_datasets, generation_identity, load_inputs,
)


class GenerationIdentityContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = normalize_config({}, self.root)
        self.windows = {
            "candidate_start_s": 0.0, "candidate_end_s": 1.0, "step_s": 1.0,
            "labels": np.asarray([1.0]), "valid": np.asarray([True]),
            "start": np.asarray([0.0]), "center": np.asarray([0.5]), "stop": np.asarray([1.0]),
            "sample_start": np.asarray([0]), "sample_stop": np.asarray([64]),
        }
        self.inputs = {"depths": np.asarray([10.0, 20.0]),
                       "range_times": np.asarray([0.0, 2.0]), "ranges": np.asarray([1.0, 2.0])}

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, path):
        return create_h5(path, "SCM", np.asarray([0]), self.windows, self.inputs,
                         np.asarray([8.0]), 3, np.asarray([0, 0, 1]), np.asarray([0, 1, 1]))

    def _write_origin_without_ctd(self, *, fs: int = 64, seconds: int = 12):
        origin = self.root / "Origindata"
        positions = origin / "positions/positions_vla.txt"
        positions.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(positions, [[1, 20.0], [2, 10.0]])
        channels = origin / "events/S5/vla_matfiles"
        channels.mkdir(parents=True, exist_ok=True)
        t = np.arange(fs * seconds, dtype=np.float64) / fs
        channel_paths = [channels / f"S5_VLA_NO_{index}.mat" for index in (1, 2)]
        for index, path in enumerate(channel_paths):
            waveform = (index + 1) * 5.0 + np.cos(2.0 * np.pi * (8.0 + index) * t)
            savemat(path, {"x": waveform})
        range_path = origin / "events/range/RangeEventS5/SproulToVLA.S5.txt"
        range_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            range_path,
            [[0, 0, 0.0, 5.0], [0, 0, 0.1, 1.0], [0, 0, 0.2, 5.0]],
            header="range",
        )
        return positions, channel_paths, range_path

    def test_report_location_and_json_order_do_not_change_identity(self):
        changed = normalize_config({"result_path": "outputs/another/report.json"}, self.root)
        reordered = {key: self.config[key] for key in reversed(self.config)}
        elsewhere = normalize_config({}, self.root / "other_project")
        for method in (None, "RBD", "SCM"):
            with self.subTest(method=method):
                self.assertEqual(generation_identity(self.config, method), generation_identity(changed, method))
                self.assertEqual(generation_identity(self.config, method), generation_identity(reordered, method))
                self.assertEqual(generation_identity(self.config, method), generation_identity(elsewhere, method))

    def test_per_method_identity_and_dataset_names_isolate_route_settings(self):
        base_cfg = normalize_config({"dataset": {"split_strategy": "range_overlap"}}, self.root)
        split_meta = {
            "left_time_start_s": 0.5,
            "right_time_start_s": 20.5,
            "left_time_end_s": 10.5,
            "right_time_end_s": 30.5,
            "range_min_time_s": 15.5,
        }
        rbd_base_id = generation_identity(base_cfg, "RBD")
        scm_base_id = generation_identity(base_cfg, "SCM")
        all_base_id = generation_identity(base_cfg)
        rbd_base_name = dataset_names("RBD", base_cfg, ["deep", "shallow"], split_meta, self.windows)[0]
        scm_base_name = dataset_names("SCM", base_cfg, ["deep", "shallow"], split_meta, self.windows)[0]

        for raw in (
            {"dataset": {"split_strategy": "range_overlap"}, "rbd": {"multipath_options": {"peak_threshold_db": -3.0}}},
            {"dataset": {"split_strategy": "range_overlap"}, "rbd": {"beam_selection": "multipath"}},
            {"dataset": {"split_strategy": "range_overlap"}, "rbd": {"use_plane_wave": True}},
            {"dataset": {"split_strategy": "range_overlap"}, "rbd": {"normalize_spectrum": True}},
            {"dataset": {"split_strategy": "range_overlap"}, "rbd": {"angle_min_deg": -60.0, "angle_max_deg": 60.0, "num_angles": 121}},
            {"dataset": {"split_strategy": "range_overlap", "ctd_file": "i9605_alt.prn"}},
        ):
            changed = normalize_config(raw, self.root)
            with self.subTest(rbd_only=raw):
                self.assertNotEqual(generation_identity(changed, "RBD"), rbd_base_id)
                self.assertNotEqual(dataset_names("RBD", changed, ["deep", "shallow"], split_meta, self.windows)[0], rbd_base_name)
                self.assertEqual(generation_identity(changed, "SCM"), scm_base_id)
                self.assertEqual(dataset_names("SCM", changed, ["deep", "shallow"], split_meta, self.windows)[0], scm_base_name)
                self.assertNotEqual(generation_identity(changed), all_base_id)

        for raw in (
            {"dataset": {"split_strategy": "range_overlap"}, "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 0}},
            {"dataset": {"split_strategy": "range_overlap"}, "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 2}},
            {"dataset": {"split_strategy": "range_overlap"}, "scm": {"norm_floor": 1e-8}},
        ):
            changed = normalize_config(raw, self.root)
            with self.subTest(scm_only=raw):
                self.assertNotEqual(generation_identity(changed, "SCM"), scm_base_id)
                self.assertNotEqual(dataset_names("SCM", changed, ["deep", "shallow"], split_meta, self.windows)[0], scm_base_name)
                self.assertEqual(generation_identity(changed, "RBD"), rbd_base_id)
                self.assertEqual(dataset_names("RBD", changed, ["deep", "shallow"], split_meta, self.windows)[0], rbd_base_name)
                self.assertNotEqual(generation_identity(changed), all_base_id)

        for raw in (
            {"dataset": {"split_strategy": "range_overlap", "event_name": "S59"}},
            {"dataset": {"split_strategy": "range_overlap", "segment_start_s": 0.25}},
            {"dataset": {"split_strategy": "range_overlap", "segment_end_s": 100.0}},
            {"dataset": {"split_strategy": "range_overlap", "segment_step_s": 0.5}},
            {"dataset": {"split_strategy": "range_overlap", "fft_window": "hann"}},
            {"dataset": {"split_strategy": "range_overlap", "signal_centering": "per_window"}},
            {"dataset": {"split_strategy": "range_overlap", "signal_centering": "none"}},
            {"dataset": {"split_strategy": "range_overlap", "frequency_selection_config": {"deep_target_freq_hz": [10.0, 20.0]}}},
        ):
            changed = normalize_config(raw, self.root)
            with self.subTest(shared=raw):
                self.assertNotEqual(generation_identity(changed, "RBD"), rbd_base_id)
                self.assertNotEqual(generation_identity(changed, "SCM"), scm_base_id)
                self.assertNotEqual(dataset_names("RBD", changed, ["deep", "shallow"], split_meta, self.windows)[0], rbd_base_name)
                self.assertNotEqual(dataset_names("SCM", changed, ["deep", "shallow"], split_meta, self.windows)[0], scm_base_name)

    def test_scm_only_load_inputs_and_generation_succeed_without_ctd_directory(self):
        self._write_origin_without_ctd(fs=64, seconds=12)
        self.assertFalse((self.root / "Origindata/environments/ctds").exists())
        loaded = load_inputs(self.root, self.config, methods=("SCM",))
        self.assertIsNone(loaded["ctd_path"])
        self.assertIsNone(loaded["profile_depths"])
        self.assertIsNone(loaded["sound_speeds"])
        self.assertTrue(all("ctds" not in item["path"] for item in loaded["input_files"]))
        self.assertTrue(np.allclose(np.mean(loaded["signal"], axis=1), 0.0))

        for mode in ("per_window", "none"):
            uncentered_cfg = normalize_config({"dataset": {"signal_centering": mode}}, self.root)
            uncentered = load_inputs(self.root, uncentered_cfg, methods=("SCM",))
            self.assertFalse(np.allclose(np.mean(uncentered["signal"], axis=1), 0.0))

        with self.assertRaises(OSError):
            load_inputs(self.root, self.config, methods=("RBD",))

        raw_cfg = {
            "schema_version": 1,
            "dataset": {
                "fs": 64,
                "segment_start_s": 0.0,
                "segment_end_s": 11.0,
                "split_strategy": "range_overlap",
                "half_duration_s": 4.0,
                "gap_s": 1.0,
                "signal_centering": "per_window",
                "fft_window": "hann",
                "frequency_selection_modes": ["deep"],
                "frequency_selection_config": {"deep_target_freq_hz": [8.0, 16.0]},
            },
            "scm": {"num_snapshots_per_segment": 2, "snapshot_overlap_count": 1, "norm_floor": 1e-10},
            "result_path": "outputs/validation/scm_only_generation.json",
        }
        report = generate_datasets(raw_cfg, self.root, methods=("SCM",))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["methods"], ["SCM"])
        self.assertEqual(set(report["datasets"]), {"SCM"})
        self.assertIsNone(report["provenance"]["ctd_source"])
        self.assertEqual(report["provenance"]["fft_window"], "hann")
        self.assertEqual(report["provenance"]["signal_centering"], "per_window")

        cfg_path = self.root / "expected_scm.json"
        cfg_path.write_text(json.dumps(raw_cfg), encoding="utf-8")
        validation_report = self.root / "outputs/validation/scm_datasets_report.json"
        code = run_datasets_command(
            root=self.root / "outputs/Datasets",
            report_path=validation_report,
            expected_config=cfg_path,
            methods=["SCM"],
        )
        self.assertEqual(code, 0)
        saved_validation = json.loads(validation_report.read_text(encoding="utf-8"))
        self.assertEqual(saved_validation["status"], "PASS")
        self.assertEqual(set(saved_validation["dataset_directories"]), {"SCM"})

    def test_loaded_input_identities_hash_actual_raw_files_and_same_stat_changes(self):
        origin = self.root / "Origindata"
        positions = origin / "positions/positions_vla.txt"
        positions.parent.mkdir(parents=True)
        np.savetxt(positions, [[1, 20], [2, 10]])
        channels = origin / "events/S5/vla_matfiles"
        channels.mkdir(parents=True)
        channel_paths = [channels / f"S5_VLA_NO_{index}.mat" for index in (1, 2)]
        for index, path in enumerate(channel_paths):
            savemat(path, {"x": np.arange(16, dtype=np.float64) + index})
        range_path = origin / "events/range/RangeEventS5/SproulToVLA.S5.txt"
        range_path.parent.mkdir(parents=True)
        np.savetxt(range_path, [[0, 0, 0, 1], [0, 0, 1, 2]], header="range")
        ctd = origin / "environments/ctds/i9605.prn"
        ctd.parent.mkdir(parents=True)
        np.savetxt(ctd, [[0, 0, 0, 1500, 0], [30, 0, 0, 1500, 0]])
        paths = [positions, *channel_paths, range_path, ctd]
        expected = [{"path": str(path), "size_bytes": path.stat().st_size,
                     "mtime_ns": path.stat().st_mtime_ns,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Raw input hashes must stream")):
            loaded = load_inputs(self.root, self.config)
        self.assertEqual(loaded["input_files"], expected)

        channel = channel_paths[0]
        stat = channel.stat()
        savemat(channel, {"x": np.arange(16, dtype=np.float64) + 0.25})
        self.assertEqual(channel.stat().st_size, stat.st_size)
        os.utime(channel, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed = load_inputs(self.root, self.config)["input_files"][1]
        self.assertEqual(changed["size_bytes"], expected[1]["size_bytes"])
        self.assertEqual(changed["mtime_ns"], expected[1]["mtime_ns"])
        self.assertNotEqual(changed["sha256"], expected[1]["sha256"])
        self.assertEqual(changed["sha256"], hashlib.sha256(channel.read_bytes()).hexdigest())

    def test_success_publishes_closed_h5_and_refuses_overwrite(self):
        path = self.root / "scm.h5"
        with self.create(path) as handle:
            handle["X"][:] = 0
            self.assertFalse(path.exists())
        self.assertTrue(path.is_file())
        with h5py.File(path, "r") as handle:
            self.assertEqual(handle["X"].shape, (1, 3, 1, 2))
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            with self.create(path):
                pass
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_failure_leaves_no_partial_final_or_temporary_file(self):
        path = self.root / "failed.h5"
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with self.create(path) as handle:
                handle["X"][:] = 0
                raise RuntimeError("intentional")
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
