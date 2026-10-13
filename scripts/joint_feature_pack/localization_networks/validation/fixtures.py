"""Small observable HDF5 fixtures for remote contract checks."""
from __future__ import annotations


def expected_pairs(method, elements):
    # Enumerate the documented MATLAB column-major upper triangle independently.
    return [(i, j) for j in range(1, elements + 1) for i in range(1, j + (method == "SCM"))]


def write_fixture(path, method, reverse, *, constant=False):
    import h5py
    import numpy as np
    n, elements, frequencies = 7, 4, 11
    pairs = expected_pairs(method, elements) if method != "RBD" else []
    height = len(pairs) if pairs else elements
    # Decimal place markers make sample, row, frequency, and RI axes observable.
    sample, row, freq, ri = np.indices((n, height, frequencies, 2))
    features = (sample * 10000 + row * 100 + freq * 3 + ri).astype(np.float32)
    if constant:
        features.fill(8.0)
    with h5py.File(path, "w") as h5:
        x = h5.create_dataset("X", data=features.transpose(3, 2, 1, 0) if reverse else features)
        logical = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
        x.attrs["axis_order"] = logical
        x.attrs["storage_axis_order"] = ",".join(logical.split(",")[::-1]) if reverse else logical
        h5["y_range_km"] = np.array([1, 2, 3, 4, np.nan, 6, 7], dtype=np.float32)
        h5["valid_sample"] = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        h5["array/depth_m"] = np.arange(elements) + 100.0
        h5["array/element_idx"] = np.arange(1, elements + 1)
        h5["frequency/freq_hz"] = np.arange(frequencies) * 5.0
        h5["split/source_segment_idx"] = np.arange(n) + 51
        h5["time/window_center_s"] = np.arange(n) + 70.5
        if pairs:
            h5["pair/numerator_element_idx"] = [p[0] for p in pairs]
            h5["pair/denominator_element_idx"] = [p[1] for p in pairs]
    return features, [0, 2, 3, 5, 6]
