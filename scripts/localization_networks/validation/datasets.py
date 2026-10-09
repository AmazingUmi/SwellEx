"""Generated HDF5, split, and cross-route acceptance checks."""
from __future__ import annotations
import json
import math
from pathlib import Path
import tempfile
from scripts.localization_networks.common.paths import PROJECT_ROOT as ROOT
from scripts.localization_networks.workflow.tasks import METHODS
from scripts.localization_networks.validation.reporting import require, normalize_expected
from scripts.localization_networks.validation.fixtures import expected_pairs, write_fixture


def vector(h5, key, *, count=None, finite=True):
    import numpy as np
    require(key in h5, f"missing {key}")
    values = np.asarray(h5[key]).reshape(-1)
    require(count is None or len(values) == count, f"{key} has {len(values)} entries; expected {count}")
    require(not finite or bool(np.isfinite(values).all()), f"{key} contains nonfinite values")
    return values


def attribute_text(value):
    import numpy as np
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return attribute_text(value.reshape(-1)[0])
    return str(value)


def classify(path):
    import h5py
    # Names take priority so malformed SCM diagonal metadata cannot be reclassified as ELM.
    for method in METHODS:
        if path.name.upper().startswith(method + "_"):
            return method
    with h5py.File(path, "r") as h5:
        if "/rbd" in h5:
            return "RBD"
        if "/pair/numerator_element_idx" in h5 and "/pair/denominator_element_idx" in h5:
            i = vector(h5, "/pair/numerator_element_idx")
            j = vector(h5, "/pair/denominator_element_idx")
            return "SCM" if (i == j).any() else "ELM"
    raise AssertionError(f"cannot identify feature method for {path}")


def read_dataset_metadata(path):
    candidates = list(path.parent.glob("*_metadata.json"))
    require(len(candidates) <= 1, f"ambiguous dataset metadata in {path.parent}")
    if not candidates:
        return {}
    value = json.loads(candidates[0].read_text(encoding="utf-8"))
    require(isinstance(value, dict), "dataset metadata must be an object")
    return value


def configured_window_check(record, expected, metadata):
    import numpy as np
    if "dataset" not in expected:
        return {}
    d, method = expected["dataset"], record["method"]
    snapshots = 1 if method == "RBD" else d["num_snapshots_per_segment"]
    hop = 1 if method == "RBD" else snapshots - d["snapshot_overlap_count"]
    base_start = d["segment_start_s"] or 0.0
    step = d["segment_step_s"] * hop
    duration = d["segment_duration_s"] + (snapshots - 1) * d["segment_step_s"]
    wanted_starts = base_start + (record["sources"].astype(np.float64) - 1) * step
    require(np.allclose(record["starts"], wanted_starts, rtol=0, atol=1e-8), "source indices/window starts disagree with configured start/step/Ns")
    require(np.allclose(record["stops"] - record["starts"], duration, rtol=0, atol=1e-8), "window duration differs from configured duration/step/Ns")
    if d["segment_end_s"] is not None:
        require(bool((record["starts"] + (snapshots - 1) * d["segment_step_s"] <= d["segment_end_s"] + 1e-8).all()), "windows exceed configured final base-window start")
    wanted_first = np.floor(record["starts"] * d["fs"] + 0.5).astype(np.uint64) + 1
    samples = round(d["fs"] * d["segment_duration_s"]) + (snapshots - 1) * round(d["fs"] * d["segment_step_s"])
    require(np.array_equal(record["sample_starts"], wanted_first), "sample_start_idx disagrees with configured fs and window time")
    require(np.array_equal(record["sample_stops"], wanted_first + samples - 1), "sample_stop_idx disagrees with configured fs/duration/Ns")
    fft_samples = round(d["fs"] * d["segment_duration_s"])
    full = np.arange(fft_samples // 2 + 1) * d["fs"] / fft_samples
    bins = record["freq"] * fft_samples / d["fs"]
    require(bool(((bins >= 0) & (bins <= fft_samples // 2)).all()) and np.allclose(bins, np.round(bins), rtol=0, atol=1e-8), "frequency axis is not on configured one-sided snapshot FFT grid")
    modes = ["full"] if method == "RBD" and expected["rbd"]["frequency_estimation"] == "full" else d["frequency_selection_modes"]
    f = d["frequency_selection_config"]
    fixed_bins = set()
    for mode in modes:
        if mode == "full":
            fixed_bins.update(range(len(full)))
        elif mode in ("deep", "shallow"):
            defaults = [49,64,79,94,112,130,148,166,201,235,283,338,388] if mode == "deep" else [109,127,145,163,198,232,280,335,385]
            targets = np.asarray(f.get(mode + "_target_freq_hz", defaults)).reshape(-1)
            fixed_bins.update(int(np.abs(full - target).argmin()) for target in targets)
        elif mode == "mel":
            lo, hi = f.get("mel_min_freq_hz", max(1, full[0])), f.get("mel_max_freq_hz", full[-1])
            mel = np.linspace(2595 * np.log10(1 + lo / 700), 2595 * np.log10(1 + hi / 700), f.get("mel_num_bins", 64))
            targets = 700 * (10 ** (mel / 2595) - 1)
            fixed_bins.update(int(np.abs(full - target).argmin()) for target in targets)
    actual_bins = set(map(int, np.round(bins)))
    if "adapt" not in modes:
        require(actual_bins == fixed_bins, "frequencies disagree with configured frequency selection")
    else:
        require(fixed_bins <= actual_bins, "adaptive union omitted configured fixed frequency bins")
        extras = full[list(actual_bins - fixed_bins)]
        require(len(extras) <= f.get("adapt_num_bins", 16) and bool(((extras >= f.get("adapt_min_freq_hz", 1)) & (extras <= f.get("adapt_max_freq_hz", full[-1])) & (extras > 0)).all()), "adaptive bins violate configured count/range")
    if metadata:
        for key, value in (("fs_hz", d["fs"]), ("segment_duration_s", duration), ("segment_step_s", step)):
            require(key in metadata and math.isclose(metadata[key], value, rel_tol=0, abs_tol=1e-8), f"metadata {key} differs from expected config")
        require(metadata.get("split_strategy") == d["split_strategy"], "metadata split strategy differs from expected config")
        if method == "RBD":
            actual = metadata.get("rbd_config", {})
            for key in ("normalize_spectrum", "use_plane_wave"):
                require(actual.get(key) == expected["rbd"][key], f"metadata RBD {key} differs from expected config")
            require(actual.get("rbd_beam_selection") == expected["rbd"]["beam_selection"], "metadata RBD beam mode differs")
            require(actual.get("rbd_frequency_estimation") == expected["rbd"]["frequency_estimation"], "metadata RBD frequency mode differs")
            if expected["rbd"]["beam_selection"] == "multipath":
                require(actual.get("rbd_multipath_options") == expected["rbd"]["multipath_options"], "metadata multipath parameters differ")
    prefix = {"RBD": "RBD_green_freq_nn_", "ELM": "ELM_lsr_freq_nn_", "SCM": "SCM_freq_nn_"}[method]
    name = Path(record["path"]).name
    if name.startswith(prefix):
        require(name.startswith(prefix + d["event_name"] + "_"), "HDF5 event name differs from expected config")
    return {"fs": d["fs"], "window_duration_s": duration, "window_step_s": step, "snapshots": snapshots, "frequency_selection": modes}


def configured_split_check(records, expected):
    import numpy as np
    if "dataset" not in expected:
        return {}
    d = expected["dataset"]
    first = records[0]
    metadata = first.get("metadata", {})
    start = d["segment_start_s"] or 0.0
    end = metadata.get("candidate_segment_end_s", d["segment_end_s"])
    require(end is not None, "null segment_end_s requires generated candidate_segment_end_s metadata")
    require(d["segment_end_s"] is None or end <= d["segment_end_s"] + 1e-8, "metadata candidate end exceeds expected config")
    base_count = int(math.floor((end - start) / d["segment_step_s"] + 1e-8)) + 1
    snapshots = 1 if first["method"] == "RBD" else d["num_snapshots_per_segment"]
    hop = 1 if first["method"] == "RBD" else snapshots - d["snapshot_overlap_count"]
    count = (base_count - snapshots) // hop + 1
    require(count > 1, "configured candidate grid has too few windows")
    if metadata:
        require(metadata.get("global_num_segments") == count, "metadata global_num_segments differs from configured grid")
    grid = np.arange(1, count + 1)
    step = hop * d["segment_step_s"]
    duration = d["segment_duration_s"] + (snapshots - 1) * d["segment_step_s"]
    overlap = {}
    if d["split_strategy"] == "full_interleaved":
        ratio = d["train_test_ratio"]
        train = grid[((grid - 1) % sum(ratio)) < ratio[0]]
        test = grid[((grid - 1) % sum(ratio)) >= ratio[0]]
        if not len(test):
            train, test = grid[:-1], grid[-1:]
    else:
        centers = start + (grid - 1) * step + duration / 2
        raw_t, raw_r = first["raw_times"], first["raw_ranges"]
        candidate_labels = np.interp(centers, raw_t, raw_r, left=np.nan, right=np.nan)
        valid = np.isfinite(candidate_labels)
        require(valid.any(), "no finite range labels on configured grid")
        pivot = int(grid[np.nanargmin(candidate_labels)])
        gap = math.floor(d["gap_s"] / step)
        width = max(1, math.floor(d["half_duration_s"] / step))
        before = grid[valid & (grid >= pivot-gap-width) & (grid < pivot-gap)]
        after = grid[valid & (grid > pivot+gap) & (grid <= pivot+gap+width)]
        require(len(before) > 0 and len(after) > 0, "range_overlap requires valid windows on both sides")
        low = max(candidate_labels[before-1].min(), candidate_labels[after-1].min())
        high = min(candidate_labels[before-1].max(), candidate_labels[after-1].max())
        require(low < high, "range_overlap has no shared range interval of positive width")
        before = before[(candidate_labels[before-1] >= low) & (candidate_labels[before-1] <= high)]
        after = after[(candidate_labels[after-1] >= low) & (candidate_labels[after-1] <= high)]
        require(len(before) > 0 and len(after) > 0, "range_overlap has no sampled windows in the shared interval")
        overlap = {"range_overlap_min_km": float(low), "range_overlap_max_km": float(high),
                   "range_overlap_num_segments_before": len(before), "range_overlap_num_segments_after": len(after)}
        if metadata:
            saved = metadata.get("split", {})
            for key, value in overlap.items():
                require(key in saved and math.isclose(saved[key], value, rel_tol=0, abs_tol=1e-10), f"metadata {key} differs from the raw-track overlap")
        train, test = (before, after) if d["train_side"] == "before" else (after, before)
        stride = max(1, math.floor((d["test_step_s"] or step) / step + 0.5))
        test = test[::stride]
    for split, wanted in (("train", train), ("test", test)):
        actual = sorted(int(source) for r in records if r["split"] == split for source in r["sources"])
        require(actual == wanted.tolist(), f"{split} source membership/coverage differs from configured {d['split_strategy']} split; expected={wanted[:20].tolist()}, actual={actual[:20]}")
    return {"strategy": d["split_strategy"], "candidate_windows": count, "expected_train": len(train), "expected_test": len(test), **overlap}


def inspect_h5(path, method, expected):
    import h5py
    import numpy as np
    metadata = read_dataset_metadata(path)
    configured_rbd = expected.get("rbd", {})
    saved_rbd = metadata.get("rbd_config", {})
    beam_mode = configured_rbd.get("beam_selection", saved_rbd.get("rbd_beam_selection"))
    frequency_mode = configured_rbd.get("frequency_estimation", saved_rbd.get("rbd_frequency_estimation"))
    norm_summary = {"mode": beam_mode, "active_columns": 0, "zero_columns": 0, "maximum_norm": 0.0,
                    "absolute_tolerance": 2e-6, "zero_threshold": 1e-8, "full_dc_exempt": frequency_mode == "full"}
    with h5py.File(path, "r") as h5:
        labels = vector(h5, "/y_range_km", finite=False)
        n = len(labels)
        require(n > 0, "empty generated split")
        valid = vector(h5, "/valid_sample", count=n)
        require(bool(np.isin(valid, [0, 1]).all()), "valid_sample must be binary")
        require(bool(np.isfinite(labels[valid.astype(bool)]).all()), "valid labels nonfinite")
        require(bool((labels[valid.astype(bool)] >= 0).all()), "valid labels negative")
        # Generation splits contain valid windows only; invalid labels must never enter acceptance data.
        require(bool(np.isfinite(labels).all()), "generated labels contain nonfinite values")
        require(bool(valid.any()), "no valid samples")
        depth = vector(h5, "/array/depth_m")
        elements = vector(h5, "/array/element_idx", count=len(depth))
        require(np.array_equal(elements, np.arange(1, len(depth) + 1)), "element indices must be 1-based ordered")
        frequencies = vector(h5, "/frequency/freq_hz")
        require(len(frequencies) > 0 and frequencies[0] >= 0 and bool((np.diff(frequencies) > 0).all()), "frequencies must be nonnegative, unique and ascending")
        height = len(depth)
        if method != "RBD":
            pairs = expected_pairs(method, len(depth))
            i = vector(h5, "/pair/numerator_element_idx", count=len(pairs))
            j = vector(h5, "/pair/denominator_element_idx", count=len(pairs))
            require(np.array_equal(np.column_stack((i, j)), np.array(pairs)), f"{method} pair order/semantics differ from column-major {'i<j' if method == 'ELM' else 'i<=j'}")
            height = len(pairs)
        require("/X" in h5, "missing /X")
        x = h5["/X"]
        stored_shape = tuple(x.shape)
        logical_axes = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
        require("axis_order" in x.attrs, "missing /X axis_order attribute")
        require(attribute_text(x.attrs["axis_order"]) == logical_axes, f"/X axis_order must be {logical_axes!r}")
        logical_shape = (n, height, len(frequencies), 2)
        if "storage_axis_order" in x.attrs:
            stored_axes = tuple(attribute_text(x.attrs["storage_axis_order"]).split(","))
            declared_axes = tuple(logical_axes.split(","))
            require(len(stored_axes) == 4 and set(stored_axes) == set(declared_axes), "invalid /X storage_axis_order")
            axes = tuple(stored_axes.index(axis) for axis in declared_axes)
            require(axes in ((0, 1, 2, 3), (3, 2, 1, 0)), "generated /X storage order must be native or MATLAB-reversed")
            require(tuple(x.shape[a] for a in axes) == logical_shape, "stored /X shape disagrees with storage_axis_order and common metadata")
            storage = "native" if axes == (0, 1, 2, 3) else "MATLAB-reversed"
        elif x.shape == logical_shape:
            axes = (0, 1, 2, 3)
            storage = "native"
        elif x.shape == logical_shape[::-1]:
            axes = (3, 2, 1, 0)
            storage = "MATLAB-reversed"
        else:
            raise AssertionError(f"/X shape {x.shape} differs from logical {logical_shape} and MATLAB reversed {logical_shape[::-1]}")
        require(x.dtype.kind == "f", f"/X must be floating-point, got {x.dtype}")
        # Read a bounded number of samples per block, covering every stored value.
        bytes_per_sample = max(1, height * len(frequencies) * 2 * x.dtype.itemsize)
        block = max(1, (16 * 1024 * 1024) // bytes_per_sample)
        for start in range(0, n, block):
            selection = [slice(None)] * 4
            selection[axes[0]] = slice(start, min(n, start + block))
            raw = np.asarray(x[tuple(selection)])
            require(bool(np.isfinite(raw).all()), f"nonfinite /X features in sample block [{start},{min(n, start + block)})")
            if method == "RBD" and beam_mode in ("best", "multipath"):
                logical = np.transpose(raw, axes).astype(np.float64)
                norms = np.sqrt(np.sum(logical ** 2, axis=(1, 3)))
                active = norms > 1e-8
                norm_summary["zero_columns"] += int((~active).sum())
                norm_summary["maximum_norm"] = max(norm_summary["maximum_norm"], float(norms.max()))
                if frequency_mode == "full":
                    active[:, frequencies == 0] = False
                norm_summary["active_columns"] += int(active.sum())
                if beam_mode == "best":
                    require(np.allclose(norms[active], 1.0, rtol=0, atol=2e-6), f"RBD best Green column norm must be one; FFT amplitude may have been saved (block {start}, maximum norm {float(norms.max()):.8g})")
                else:
                    require(bool((norms <= 1 + 2e-6).all()), f"RBD multipath Green column norm exceeds one; FFT amplitude may have been saved (block {start})")
        starts = vector(h5, "/time/window_start_s", count=n)
        centers = vector(h5, "/time/window_center_s", count=n)
        stops = vector(h5, "/time/window_stop_s", count=n)
        require(bool(((starts <= centers) & (centers <= stops) & (starts < stops)).all()), "invalid window start/center/stop ordering")
        require(np.allclose(centers, (starts + stops) / 2, rtol=0, atol=1e-8), "window center is not midpoint")
        sample_starts = vector(h5, "/time/sample_start_idx", count=n)
        sample_stops = vector(h5, "/time/sample_stop_idx", count=n)
        sources = vector(h5, "/split/source_segment_idx", count=n)
        for key, values in (("sample_start_idx", sample_starts), ("sample_stop_idx", sample_stops), ("source_segment_idx", sources)):
            require(bool(((values >= 1) & (values == np.floor(values))).all()), f"{key} must be positive integer indices")
        require(bool((sample_starts <= sample_stops).all()), "sample windows have reversed endpoints")
        require(len(set(map(int, sources))) == n, "duplicate source_segment_idx within split")
        require(bool((np.diff(sources.astype(np.int64)) > 0).all()), "source_segment_idx must be strictly increasing")
        require(bool((np.diff(starts) > 0).all()) and bool((np.diff(sample_starts.astype(np.int64)) > 0).all()), "window source order not increasing")
        raw_times = vector(h5, "/range/range_time_s")
        raw_ranges = vector(h5, "/range/range_km_raw", count=len(raw_times))
        require(len(raw_times) > 1 and bool((np.diff(raw_times) > 0).all()), "raw range time must be strictly increasing")
        require(bool((raw_ranges >= 0).all()), "raw ranges negative")
        require(bool(((centers >= raw_times[0]) & (centers <= raw_times[-1])).all()), "window label times outside raw range coverage")
        interpolated = np.interp(centers, raw_times, raw_ranges)
        require(np.allclose(labels, interpolated, rtol=1e-6, atol=1e-6), "range labels disagree with linear interpolation of raw range at window centers")
        if "num_elements" in expected:
            require(len(depth) == expected["num_elements"], "num_elements differs from expected config")
        if "freq_hz" in expected:
            require(np.allclose(frequencies, expected["freq_hz"], rtol=0, atol=1e-8), "frequency axis differs from expected config")
        require(n >= expected.get("min_samples", 1), "split below expected min_samples")
    split = next((s for s in ("train", "val", "test") if path.stem.endswith("_" + s)), None)
    require(split is not None, f"generated HDF5 filename must end in _train/_val/_test: {path.name}")
    record = {"path": str(path), "method": method, "directory": str(path.parent), "split": split,
              "labels": labels, "valid": valid, "freq": frequencies, "depth": depth, "sources": sources,
              "starts": starts, "centers": centers, "stops": stops, "sample_starts": sample_starts, "sample_stops": sample_stops,
              "raw_times": raw_times, "raw_ranges": raw_ranges, "metadata": metadata}
    configured_details = configured_window_check(record, expected, metadata)
    details = {"path": str(path), "method": method, "split": split, "samples": n, "valid_samples": int(valid.sum()),
               "logical_shape": list(logical_shape), "stored_shape": list(stored_shape), "storage": storage,
               "frequency_hz": frequencies.tolist(), "source_segment_idx_min": int(sources.min()), "source_segment_idx_max": int(sources.max())}
    if method == "RBD" and beam_mode:
        details["green_column_norm"] = norm_summary
    if configured_details:
        details["expected_config"] = configured_details
    return record, details


def split_check(records, expected=None):
    import numpy as np
    require(any(r["split"] == "train" for r in records), "missing train split")
    require(any(r["split"] == "test" for r in records), "missing test split")
    for a_idx, a in enumerate(records):
        for b in records[a_idx + 1:]:
            require(np.array_equal(a["freq"], b["freq"]), "frequency axis differs among files in dataset directory")
            require(np.array_equal(a["depth"], b["depth"]), "array depth differs among files in dataset directory")
            if a["split"] == b["split"]:
                continue
            overlap = set(map(int, a["sources"])) & set(map(int, b["sources"]))
            require(not overlap, f"{a['split']}/{b['split']} source windows overlap: {sorted(overlap)[:10]}")
            intervals = sorted([(int(s), int(e), a["split"]) for s, e in zip(a["sample_starts"], a["sample_stops"])]+
                               [(int(s), int(e), b["split"]) for s, e in zip(b["sample_starts"], b["sample_stops"])])
            maxima = {}
            for start, stop, split in intervals:
                other = b["split"] if split == a["split"] else a["split"]
                require(maxima.get(other, 0) < start, f"{a['split']}/{b['split']} contain overlapping original sample intervals near {start}")
                maxima[split] = max(maxima.get(split, 0), stop)
    configured = configured_split_check(records, expected or {})
    return {"files": [r["path"] for r in records], "source_windows": "disjoint", "original_sample_intervals": "disjoint", "expected_config": configured}


def crossroute_check(records):
    import numpy as np
    groups = {}
    for record in records:
        groups.setdefault((record["method"], record["directory"]), []).append(record)
    require(set(m for m, _ in groups) == set(METHODS), "all RBD, ELM, SCM routes are required")
    def signature(group):
        return sorted((r["split"], tuple(r["sources"].tolist()), tuple(r["sample_starts"].tolist()), tuple(r["sample_stops"].tolist()), tuple(r["freq"].tolist())) for r in group)
    matched = []
    for (method, directory), group in groups.items():
        peers = []
        for other in METHODS:
            if other == method:
                continue
            candidates = [(d, g) for (m, d), g in groups.items() if m == other and signature(g) == signature(group)]
            require(candidates, f"{directory}: no {other} route shares splits, source windows and frequencies")
            for peer_directory, peer in candidates:
                for r in group:
                    corresponding = [p for p in peer if p["split"] == r["split"] and np.array_equal(p["sources"], r["sources"])]
                    require(len(corresponding) == 1, "ambiguous corresponding HDF5 split")
                    p = corresponding[0]
                    for key in ("labels", "valid", "starts", "centers", "stops", "depth"):
                        require(np.allclose(r[key], p[key], rtol=1e-6, atol=1e-7), f"{directory} vs {peer_directory}: common {key} differs")
                peers.append(peer_directory)
        matched.append({"method": method, "directory": directory, "matching_routes": sorted(set(peers))})
    return {"matched_datasets": matched}


def validator_controls_check():
    """Acceptance data must pass, and deliberately corrupted contracts must fail."""
    import h5py
    import numpy as np
    def rejected(function, expected_text):
        try:
            function()
        except AssertionError as exc:
            require(expected_text.casefold() in str(exc).casefold(), f"wrong rejection: {exc}; expected {expected_text}")
        else:
            raise AssertionError(f"corrupted fixture was accepted; expected {expected_text}")
    with tempfile.TemporaryDirectory(prefix="swellex-validator-") as tmp:
        directory = Path(tmp)
        records = []
        fixtures = {}
        for method in METHODS:
            route = directory / method
            route.mkdir()
            for split, offset in (("train", 1), ("test", 11)):
                path = route / f"{method}_fixture_{split}.h5"
                write_fixture(path, method, False)
                sources = np.arange(7) + offset
                starts = sources.astype(np.float64)
                with h5py.File(path, "r+") as h5:
                    h5["y_range_km"][...] = starts + 1.5
                    h5["valid_sample"][...] = np.ones(7)
                    h5["split/source_segment_idx"][...] = sources
                    h5["time/window_center_s"][...] = starts + 0.5
                    h5["time/window_start_s"] = starts
                    h5["time/window_stop_s"] = starts + 1.0
                    h5["time/sample_start_idx"] = sources * 100 + 1
                    h5["time/sample_stop_idx"] = sources * 100 + 100
                    h5["range/range_time_s"] = np.array([0.0, 100.0])
                    h5["range/range_km_raw"] = np.array([1.0, 101.0])
                record, _ = inspect_h5(path, method, {})
                records.append(record)
                fixtures[(method, split)] = path
            split_check([r for r in records if r["method"] == method])
        crossroute_check(records)
        path = fixtures[("RBD", "train")]
        with h5py.File(path, "r+") as h5:
            h5["X"][0, 0, 0, 0] = np.nan
        rejected(lambda: inspect_h5(path, "RBD", {}), "nonfinite /X")
        with h5py.File(path, "r+") as h5:
            h5["X"][0, 0, 0, 0] = 0.0
            del h5["X"].attrs["axis_order"]
        rejected(lambda: inspect_h5(path, "RBD", {}), "axis_order")
        path = fixtures[("ELM", "train")]
        with h5py.File(path, "r+") as h5:
            for key in ("pair/numerator_element_idx", "pair/denominator_element_idx"):
                h5[key][...] = np.asarray(h5[key])[::-1]
        rejected(lambda: inspect_h5(path, "ELM", {}), "pair order/semantics")
        path = fixtures[("SCM", "train")]
        with h5py.File(path, "r+") as h5:
            h5["pair/numerator_element_idx"][0] = 2
        require(classify(path) == "SCM", "corrupt SCM metadata was misclassified")
        rejected(lambda: inspect_h5(path, "SCM", {}), "pair order/semantics")
        rbd = [r for r in records if r["method"] == "RBD"]
        corrupted = dict(rbd[1])
        corrupted["sample_starts"] = corrupted["sample_starts"].copy()
        corrupted["sample_stops"] = corrupted["sample_stops"].copy()
        corrupted["sample_starts"][0] = rbd[0]["sample_starts"][-1]
        corrupted["sample_stops"][0] = rbd[0]["sample_stops"][-1]
        rejected(lambda: split_check([rbd[0], corrupted]), "overlapping original sample intervals")
        changed_label = [dict(r) for r in records]
        changed_label[-1]["labels"] = changed_label[-1]["labels"] + 1.0
        rejected(lambda: crossroute_check(changed_label), "common labels differs")
        # A separate physical RBD fixture distinguishes normalized Green columns from raw FFT amplitudes.
        rbd_path = fixtures[("RBD", "train")]
        canonical = normalize_expected({"schema_version": 1, "dataset": {"fs": 100, "segment_start_s": 1,
            "segment_end_s": 17, "frequency_selection_modes": ["deep"],
            "frequency_selection_config": {"deep_target_freq_hz": list(range(0, 51, 5))}},
            "rbd": {"beam_selection": "best"}, "result_path": "outputs/validation/control.json"})
        with h5py.File(rbd_path, "r+") as h5:
            physical = np.zeros_like(h5["X"])
            physical[..., 0] = 0.5  # Four elements: sum |g|^2 = 1.
            physical[:, :, 0, :] = 0.0  # Truly zero columns remain permitted.
            h5["X"][...] = physical
            h5["X"].attrs["axis_order"] = "sample,element,frequency,real_imag"
        inspect_h5(rbd_path, "RBD", canonical)
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 16.0
        rejected(lambda: inspect_h5(rbd_path, "RBD", canonical), "RBD best Green column norm")
        multipath = normalize_expected({**canonical, "rbd": {**canonical["rbd"], "beam_selection": "multipath"}})
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 0.5
        inspect_h5(rbd_path, "RBD", multipath)
        rejected(lambda: inspect_h5(rbd_path, "RBD", canonical), "RBD best Green column norm")
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 2.0
        rejected(lambda: inspect_h5(rbd_path, "RBD", multipath), "RBD multipath Green column norm")
        rejected(lambda: normalize_expected({"dataset": {"segment_typo_s": 1}}), "unknown dataset fields")
        rejected(lambda: normalize_expected({"rbd": {"beam_selecton": "best"}}), "unknown rbd fields")
    return {"positive_controls": "three complete routes accepted", "rejected_corruptions": ["nonfinite features", "missing axis metadata",
            "ELM pair order", "SCM pair semantics", "split raw sample leakage", "cross-route label inconsistency",
            "FFT amplitude instead of normalized Green", "multipath norm above one", "unknown canonical config fields"]}


def datasets(report, root, expected):
    paths = sorted(p for p in root.rglob("*.h5") if not p.name.startswith("._"))
    report.check("datasets.discovery", lambda: require(paths, f"no generated HDF5 files under {root}") or {"root": str(root), "files": len(paths)})
    records = []
    for path in paths:
        def inspect(p=path):
            method = classify(p)
            directory = str(p.parent)
            if directory not in report.data["dataset_directories"][method]:
                report.data["dataset_directories"][method].append(directory)
            record, details = inspect_h5(p, method, expected)
            records.append(record)
            return details
        report.check(f"dataset.file.{path.relative_to(root)}", inspect)
    groups = {}
    for record in records:
        groups.setdefault((record["method"], record["directory"]), []).append(record)
    for (method, directory), group in sorted(groups.items()):
        report.check(f"dataset.splits.{method}.{Path(directory).name}", lambda g=group: split_check(g, expected))
    report.check("datasets.cross_route_consistency", lambda: crossroute_check(records))
    report.data["dataset_directories"] = {m: sorted(v) for m, v in report.data["dataset_directories"].items()}
    report.data["datasets"] = {m: dirs[0] for m, dirs in report.data["dataset_directories"].items() if len(dirs) == 1}
