"""Verify MoE_learn per-window centering never leaks across windows or raw inputs."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.data_processing.MoE_learn.generation_core import centered_window, load_inputs


class WindowCenteringTests(unittest.TestCase):
    def test_window_centering_ignores_other_windows_and_preserves_raw_inputs(self):
        signal = np.arange(60, dtype=np.float64).reshape(3, 20)
        original = signal.copy()
        expected = centered_window(signal, 5, 10)
        np.testing.assert_array_equal(signal, original)
        signal[:, :5] += 10000
        signal[:, 10:] -= 10000
        np.testing.assert_array_equal(expected, centered_window(signal, 5, 10))
        np.testing.assert_allclose(expected.mean(axis=1), 0)
        # Guard the input-loading boundary too: no global centering before extraction.
        positions = np.array([[1., 100.], [2., 120.]])
        ranges = np.array([[0., 1.], [1., 2.]])
        ctd = np.array([[0., 0., 0., 1500., 0.], [200., 0., 0., 1500., 0.]])
        channels = [np.arange(20, dtype=np.float64) + offset for offset in (100, 200)]
        with patch("scripts.data_processing.MoE_learn.generation_core.np.loadtxt", side_effect=[positions, ranges, ctd]), \
                patch("scripts.data_processing.MoE_learn.generation_core.load_channel", side_effect=channels), \
                patch("scripts.data_processing.MoE_learn.generation_core._input_file_identity", return_value={}):
            loaded = load_inputs(Path("/synthetic"), {"dataset": {"event_name": "S5"}})
        np.testing.assert_array_equal(loaded["signal"], np.stack(channels))


if __name__ == "__main__":
    unittest.main()
