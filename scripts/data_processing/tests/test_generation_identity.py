"""Generation identity and atomic output ownership; execute on haiqin."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.data_processing.generation_config import normalize_config
from scripts.data_processing.generation_core import create_h5, dataset_names, generation_identity


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
