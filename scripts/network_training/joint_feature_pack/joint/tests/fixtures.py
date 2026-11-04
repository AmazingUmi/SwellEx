"""Small, aligned native route datasets with complete provenance."""
from pathlib import Path
import json

import h5py
import numpy as np


def write_routes(root: Path) -> dict[str, Path]:
    result = {}
    for method in ("RBD", "ELM", "SCM"):
        directory = root / method
        directory.mkdir(parents=True)
        result[method] = directory
        elements, frequency = 3, 4
        pairs = [(i, j) for j in range(1, elements + 1) for i in range(1, j + (method == "SCM"))]
        height = elements if method == "RBD" else len(pairs)
        scale = {"RBD": 0.2, "ELM": 20.0, "SCM": 0.05}[method]
        for split, count, offset in (("train", 8, 0), ("test", 3, 8)):
            with h5py.File(directory / f"{method}_{split}.h5", "w") as handle:
                values = (1 + np.arange(count * height * frequency * 2, dtype=np.float32).reshape(count, height, frequency, 2)) * scale
                x = handle.create_dataset("X", data=values)
                order = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
                x.attrs["axis_order"] = x.attrs["storage_axis_order"] = order
                handle.attrs["generator_version"] = "fixture-v1"
                times = np.arange(count, dtype=np.float64) + offset
                fields = {"y_range_km": 1 + times / 10, "valid_sample": np.ones(count, dtype=np.uint8),
                          "split/source_segment_idx": times.astype(np.uint64) + 1,
                          "time/window_start_s": times, "time/window_center_s": times + .5,
                          "time/window_stop_s": times + 1,
                          "time/sample_start_idx": times.astype(np.uint64) * 10 + 1,
                          "time/sample_stop_idx": (times.astype(np.uint64) + 1) * 10,
                          "frequency/freq_hz": [10., 20., 30., 40.], "array/element_idx": [1, 2, 3],
                          "array/depth_m": [10., 20., 30.], "range/range_time_s": np.arange(12),
                          "range/range_km_raw": 1 + np.arange(12) / 10}
                if method != "RBD":
                    fields.update({"pair/numerator_element_idx": [pair[0] for pair in pairs],
                                   "pair/denominator_element_idx": [pair[1] for pair in pairs]})
                for key, values in fields.items():
                    handle.create_dataset(key, data=np.asarray(values).reshape(-1, 1))
        metadata = {"input_files": [{"path": "/raw/recording.mat", "size_bytes": 10, "mtime_ns": 100,
                                      "sha256": "a" * 64}],
                    "effective_config": {"schema_version": 1,
                                         "dataset": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0},
                                         "rbd": {"beam_selection": "best"}},
                    "split": {"strategy": "full_interleaved", "train_count": 8, "test_count": 3},
                    "generator_version": "fixture-v1"}
        (directory / f"{method}_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return result
