"""Generation identity and atomic output ownership; execute on haiqin."""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from scipy.io import savemat

from scripts.joint_feature_pack.data_processing.generation_config import normalize_config
from scripts.joint_feature_pack.data_processing.generation_core import create_h5, dataset_names, generation_identity, load_inputs


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

    def test_report_location_and_json_order_do_not_change_identity(self):
        changed = normalize_config({"result_path": "outputs/another/report.json"}, self.root)
        self.assertEqual(generation_identity(self.config), generation_identity(changed))
        reordered = {key: self.config[key] for key in reversed(self.config)}
        self.assertEqual(generation_identity(self.config), generation_identity(reordered))
        elsewhere = normalize_config({}, self.root / "other_project")
        self.assertEqual(generation_identity(self.config), generation_identity(elsewhere))

    def test_semantically_different_configurations_have_distinct_directories(self):
        baseline = dataset_names("SCM", self.config, ["deep", "shallow"], {}, self.windows)[0]
        for raw in ({"dataset": {"event_name": "S59"}},
                    {"dataset": {"segment_start_s": 0.25}},
                    {"dataset": {"segment_end_s": 100.0}},
                    {"dataset": {"segment_step_s": 0.5}},
                    {"dataset": {"frequency_selection_config": {"deep_target_freq_hz": [10.0, 20.0]}}},
                    {"rbd": {"multipath_options": {"peak_threshold_db": -3.0}}}):
            changed = normalize_config(raw, self.root)
            with self.subTest(raw=raw):
                name = dataset_names("SCM", changed, ["deep", "shallow"], {}, self.windows)[0]
                self.assertNotEqual(name, baseline)

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
