"""NumPy translations of the reviewed DS/SPL/RBD/ELM/SCM formulas.

Indices inside Python are zero based. Persisted FFT/element/source/sample
indices are one based, matching the HDF5 research contract.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

EPS = np.finfo(np.float64).eps
DEEP = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
SHALLOW = [109, 127, 145, 163, 198, 232, 280, 335, 385]


def round_positive(value):
    return np.floor(np.asarray(value) + 0.5).astype(np.int64)


def pair_indices(elements, diagonal=False):
    if isinstance(elements, bool) or int(elements) != elements or elements < (1 if diagonal else 2):
        raise ValueError("Invalid element count")
    pairs = [(i, j) for j in range(elements) for i in range(j + int(diagonal))]
    i, j = zip(*pairs)
    return np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)


def segments(signal, fs, duration, step, start, end, range_times, ranges):
    record_duration = signal.shape[1] / fs
    start = max(0.0, 0.0 if start is None else start)
    end = min(record_duration - duration, record_duration - duration if end is None else end)
    if end < start:
        raise ValueError("Invalid segmentation interval for signal record")
    count = int(np.floor((end-start)/step + 1e-10)) + 1
    starts = start + np.arange(count) * step
    centers = starts + duration/2
    sample_start = round_positive(starts*fs)
    sample_stop = sample_start + int(round_positive(fs*duration))
    if np.any(sample_stop > signal.shape[1]):
        raise ValueError("Segment exceeds available signal samples")
    labels = np.interp(centers, range_times, ranges, left=np.nan, right=np.nan)
    return {"start": starts, "center": centers, "stop": starts+duration,
            "sample_start": sample_start, "sample_stop": sample_stop,
            "labels": labels, "valid": np.isfinite(labels),
            "duration_s": duration, "step_s": step,
            "candidate_start_s": start, "candidate_end_s": end,
            "snapshot_num_samples": int(round_positive(fs*duration)),
            "snapshot_step_samples": int(round_positive(fs*step))}


def combine_snapshots(base, ns, overlap, range_times, ranges):
    stride = ns-overlap
    if stride < 1 or len(base["start"]) < ns:
        raise ValueError("Not enough base snapshots or invalid overlap")
    first = np.arange(0, len(base["start"])-ns+1, stride)
    last = first+ns-1
    starts, stops = base["start"][first], base["stop"][last]
    centers = (starts+stops)/2
    labels = np.interp(centers, range_times, ranges, left=np.nan, right=np.nan)
    valid = np.isfinite(labels) & np.asarray([np.all(base["valid"][i:i+ns]) for i in first])
    return base | {"start": starts, "center": centers, "stop": stops,
                   "sample_start": base["sample_start"][first], "sample_stop": base["sample_stop"][last],
                   "labels": labels, "valid": valid,
                   "duration_s": base["duration_s"]+(ns-1)*base["step_s"],
                   "step_s": stride*base["step_s"]}


def build_split(windows, config):
    n = len(windows["start"])
    meta = {"split_strategy": config["split_strategy"]}
    if config["split_strategy"] == "full_interleaved":
        ratio = config["train_test_ratio"]
        test_mask = np.arange(n) % sum(ratio) >= ratio[0]
        if not np.any(test_mask) and n > 1:
            test_mask[-1] = True
        train, test = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        meta.update(train_test_ratio=ratio, split_period=sum(ratio))
    elif config["split_strategy"] == "range_overlap":
        valid = np.flatnonzero(windows["valid"] & np.isfinite(windows["labels"]))
        if not len(valid):
            raise ValueError("range_overlap requires finite labels")
        minimum = int(valid[np.argmin(windows["labels"][valid])])
        step = windows["step_s"]
        half = max(1, int(np.floor(config["half_duration_s"]/step)))
        gap = int(np.floor(config["gap_s"]/step))
        left = valid[(valid >= max(0, minimum-gap-half)) & (valid < minimum-gap)]
        right = valid[(valid > minimum+gap) & (valid <= min(n-1, minimum+gap+half))]
        if not len(left) or not len(right):
            raise ValueError("range_overlap requires valid windows on both sides of closest approach")
        candidate_left, candidate_right = len(left), len(right)
        labels = windows["labels"]
        lower = max(float(np.min(labels[left])), float(np.min(labels[right])))
        upper = min(float(np.max(labels[left])), float(np.max(labels[right])))
        if lower >= upper:
            raise ValueError("range_overlap requires a shared range interval of positive width")
        left = left[(labels[left] >= lower) & (labels[left] <= upper)]
        right = right[(labels[right] >= lower) & (labels[right] <= upper)]
        if not len(left) or not len(right):
            raise ValueError("range_overlap has no sampled windows inside the shared range interval")
        train, test_full = (left, right) if config["train_side"] == "before" else (right, left)
        test_step = step if config["test_step_s"] is None else config["test_step_s"]
        if test_step < step:
            raise ValueError("test_step_s must not be smaller than window step")
        stride = max(1, int(round_positive(test_step/step)))
        test = test_full[::stride]
        meta.update(range_overlap_train_side=config["train_side"],
                    range_overlap_half_duration_s=config["half_duration_s"], range_overlap_gap_s=config["gap_s"],
                    range_overlap_test_step_s=test_step, range_overlap_has_custom_test_step=config["test_step_s"] is not None,
                    range_overlap_test_stride_segments=stride, range_overlap_half_num_segments=half,
                    range_overlap_gap_num_segments=gap, range_overlap_min_km=lower, range_overlap_max_km=upper,
                    range_overlap_candidate_num_segments_before=candidate_left,
                    range_overlap_candidate_num_segments_after=candidate_right,
                    range_overlap_num_segments_before=len(left), range_overlap_num_segments_after=len(right),
                    range_overlap_full_test_num_segments=len(test_full), range_min_segment_idx=minimum+1,
                    range_min_time_s=windows["center"][minimum], range_min_km=windows["labels"][minimum])
        for name, indices in (("left", left), ("right", right)):
            meta.update({f"{name}_segment_idx_first": int(indices[0])+1,
                         f"{name}_segment_idx_last": int(indices[-1])+1,
                         f"{name}_time_start_s": windows["center"][indices[0]],
                         f"{name}_time_end_s": windows["center"][indices[-1]],
                         f"{name}_range_km_min": np.min(windows["labels"][indices]),
                         f"{name}_range_km_max": np.max(windows["labels"][indices])})
    else:
        raise ValueError("split_strategy must be full_interleaved or range_overlap")
    if not len(train) or not len(test):
        raise ValueError("Split produced an empty training or test dataset")
    for name, indices in (("train", train), ("test", test)):
        meta.update({f"num_{name}_samples": len(indices), f"{name}_segment_idx_first": int(indices[0])+1,
                     f"{name}_segment_idx_last": int(indices[-1])+1})
    meta["sample_indices_dataset"] = "/split/source_segment_idx"
    return {"train": train, "test": test}, meta


def snapshot_fft(signal, samples, step, bins=None):
    if signal.ndim != 2 or signal.shape[0] < 1 or samples < 2 or step < 1 or samples > signal.shape[1]:
        raise ValueError("Invalid snapshot signal dimensions")
    starts = np.arange(0, signal.shape[1]-samples+1, step)
    # [element, frequency, snapshot], including complex input one-sided FFT.
    spectra = np.stack([np.fft.fft(signal[:, i:i+samples], axis=1)[:, :samples//2+1] for i in starts], axis=2)
    return spectra if bins is None else spectra[:, bins, :]


def elm_feature(signal, samples, step, bins, pair_i, pair_j, denominator_floor_relative=1e-6):
    spectra = snapshot_fft(signal, samples, step, bins)
    power = np.abs(spectra)**2
    finite = power[np.isfinite(power)]
    floor = denominator_floor_relative * (np.median(finite) if finite.size else 0.0)
    numerator = np.sum(spectra[pair_i] * np.conj(spectra[pair_j]), axis=2)
    denominator = np.sum(np.abs(spectra[pair_j])**2, axis=2)+floor+EPS
    return numerator/denominator, {"denom_floor": float(floor), "num_snapshots": spectra.shape[2]}


def scm_feature(signal, samples, step, bins, pair_i, pair_j, norm_floor=1e-12):
    spectra = snapshot_fft(signal, samples, step, bins)
    vectors = spectra/np.maximum(np.linalg.norm(spectra, axis=0, keepdims=True), norm_floor)
    covariance = np.einsum("ifs,jfs->ijf", vectors, vectors.conj())/vectors.shape[2]
    return covariance[pair_i, pair_j], {"num_snapshots": spectra.shape[2]}


def select_frequency_bins(full_frequencies, modes, config=None, *, signal=None, windows=None, train_indices=None):
    frequencies = np.asarray(full_frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequency axis must be finite and strictly increasing")
    if not modes or any(mode not in {"full", "mel", "deep", "shallow", "adapt"} for mode in modes):
        raise ValueError("Invalid frequency modes")
    if "full" in modes and len(modes) != 1:
        raise ValueError("full cannot be combined with other frequency modes")
    cfg = {"mel_num_bins": 64, "mel_min_freq_hz": max(1.0, frequencies[0]), "mel_max_freq_hz": frequencies[-1],
           "deep_target_freq_hz": DEEP, "shallow_target_freq_hz": SHALLOW, "adapt_num_bins": 16,
           "adapt_min_freq_hz": max(1.0, frequencies[0]), "adapt_max_freq_hz": frequencies[-1],
           "adapt_pooling": "mean_power_over_elements_snapshots_windows"} | (config or {})
    selected, info = [], []
    training_candidates = np.asarray([], dtype=np.int64)
    for mode in modes:
        requested, power, centers = [], [], []
        if mode == "full":
            bins = np.arange(len(frequencies))
        elif mode in {"deep", "shallow", "mel"}:
            if mode == "mel":
                low = max(cfg["mel_min_freq_hz"], frequencies[0]); high = min(cfg["mel_max_freq_hz"], frequencies[-1])
                if low >= high:
                    raise ValueError("Invalid Mel frequency interval")
                centers = 700*(10**(np.linspace(2595*np.log10(1+low/700), 2595*np.log10(1+high/700), int(cfg["mel_num_bins"]))/2595)-1)
                requested = centers
            else:
                requested = cfg[mode+"_target_freq_hz"]
            raw_bins = np.asarray([np.argmin(np.abs(frequencies-f)) for f in requested])
            _, positions = np.unique(raw_bins, return_index=True)
            positions = np.sort(positions)
            bins = raw_bins[positions]
            if mode == "mel":
                centers = centers[positions]
                requested = centers
        else:
            if signal is None or windows is None or train_indices is None:
                raise ValueError("Adaptive selection requires explicit training windows")
            training_candidates = np.asarray(train_indices, dtype=np.int64)
            training_candidates = training_candidates[windows["valid"][training_candidates]]
            if not len(training_candidates):
                raise ValueError("Adaptive selection requires valid training windows")
            if cfg.get("adapt_max_num_segments") and len(training_candidates) > cfg["adapt_max_num_segments"]:
                keep = round_positive(np.linspace(0, len(training_candidates)-1, cfg["adapt_max_num_segments"]))
                training_candidates = training_candidates[keep]
            sums = np.zeros(len(frequencies)); count = 0
            for index in training_candidates:
                section = signal[:, windows["sample_start"][index]:windows["sample_stop"][index]]
                spectra = snapshot_fft(section, windows["snapshot_num_samples"], windows["snapshot_step_samples"])
                sums += np.nansum(np.nanmean(np.abs(spectra)**2, axis=0), axis=1)
                count += spectra.shape[2]
            if not count:
                raise ValueError("Adaptive selection found no training snapshots")
            power = sums/count
            mask = (frequencies >= cfg["adapt_min_freq_hz"]) & (frequencies <= cfg["adapt_max_freq_hz"]) & (frequencies > 0)
            eligible = np.flatnonzero(mask)
            if not len(eligible):
                raise ValueError("Adaptive interval contains no positive frequency bins")
            order = np.argsort(-power[eligible], kind="stable")
            bins = np.sort(eligible[order[:min(int(cfg["adapt_num_bins"]), len(order))]])
        selected.extend(bins.tolist())
        info.append({"mode": mode, "requested_freq_hz": requested, "selected_fft_bin_idx": bins+1,
                     "selected_freq_hz": frequencies[bins], "mel_center_freq_hz": centers, "adapt_power": power})
    bins = np.unique(selected).astype(np.int64)
    return bins, frequencies[bins], {"frequency_selection_modes": modes, "selected_fft_bin_idx": bins+1,
             "selected_freq_hz": frequencies[bins], "num_freq_bins": len(bins), "mode_info": info,
             "has_duplicate_requested_bins": len(bins) < len(selected), "config": cfg,
             "fit_scope": "training_split_only", "adapt_training_segment_idx": training_candidates+1}


def extend_profile(depths, speeds, required_depth):
    depths, speeds = np.asarray(depths, dtype=float), np.asarray(speeds, dtype=float)
    if np.max(depths) < required_depth:
        extra = np.arange(np.max(depths)+0.5, required_depth+1e-10, 0.5)
        depths = np.concatenate([depths, extra]); speeds = np.concatenate([speeds, np.full(len(extra), speeds[-1])])
    return depths, speeds


def steering_delays(theta, array_depths, sound_speeds, profile_depths, plane_wave=False):
    theta = np.asarray(theta, dtype=float); array_depths = np.asarray(array_depths, dtype=float)
    profile_depths, sound_speeds = np.asarray(profile_depths, dtype=float), np.asarray(sound_speeds, dtype=float)
    minimum = int(np.argmin(sound_speeds)); c0, z0 = sound_speeds[minimum], profile_depths[minimum]
    plane = ((array_depths-z0)/c0)[:, None]*np.sin(theta)[None, :]
    if plane_wave:
        return plane
    delays = np.zeros_like(plane)
    for a, angle in enumerate(theta):
        ray_parameter = np.cos(angle)**2/c0**2
        fallback = False
        for e, depth in enumerate(array_depths):
            mask = (profile_depths >= min(z0, depth)) & (profile_depths <= max(z0, depth))
            z, c = profile_depths[mask], sound_speeds[mask]
            if len(z) < 2:
                continue
            integrand = 1/c**2-ray_parameter
            if np.any(integrand < 0) or not np.all(np.isfinite(integrand)):
                fallback = True; break
            sign = np.sign(np.sin(angle))*(1 if depth >= z0 else -1)
            delays[e, a] = sign*trapezoid(np.sqrt(integrand), z)
        if fallback or np.any(np.isnan(delays[:, a])):
            delays[:, a] = plane[:, a]
    return delays


def bartlett_beamformer(spectrum, omega, delays):
    beam = np.empty((spectrum.shape[1], delays.shape[1]), dtype=np.complex128)
    for a in range(delays.shape[1]):
        valid = ~np.isnan(delays[:, a])
        if not np.any(valid):
            beam[:, a] = np.nan
        else:
            beam[:, a] = np.mean(np.exp(-1j*delays[valid, a, None]*omega[None, :])*spectrum[valid], axis=0)
    return beam


def sidelobe_psf(delays, omega):
    angles = delays.shape[1]
    psf = np.zeros((angles, angles))
    for source in range(angles):
        for scan in range(angles):
            valid = ~np.isnan(delays[:, source]) & ~np.isnan(delays[:, scan])
            if not np.any(valid):
                psf[scan, source] = np.nan; continue
            difference = delays[valid, source]-delays[valid, scan]
            response = np.mean(np.exp(1j*difference[:, None]*omega[None, :]), axis=0)
            psf[scan, source] = np.sum(np.abs(response)**2)
        largest = np.max(psf[:, source])
        if np.isfinite(largest) and largest > 0:
            psf[:, source] /= largest
    return np.where(np.isfinite(psf), psf, 0.0)


def select_beam_peaks(power, theta, delays, omega, options, psf=None):
    best = int(np.argmax(power)); maximum = power[best]
    psf = sidelobe_psf(delays, omega) if psf is None else psf
    predicted = np.zeros(len(power)); rejected = []
    if len(power) == 1 or not np.isfinite(maximum):
        return np.asarray([best]), np.asarray(rejected, dtype=int), predicted, psf
    threshold = maximum*10**(options["peak_threshold_db"]/10)
    candidates = []
    for index, value in enumerate(power):
        local = ((len(power) == 1) or (index == 0 and value > power[1])
                 or (index == len(power)-1 and value > power[-2])
                 or (0 < index < len(power)-1 and value >= power[index-1] and value > power[index+1]))
        if np.isfinite(value) and value >= threshold and local:
            candidates.append(index)
    candidates.sort(key=lambda index: -power[index])
    selected = []
    for index in candidates:
        if selected and np.any(np.abs(theta[index]-theta[selected]) < np.deg2rad(options["min_separation_deg"])):
            continue
        if selected:
            prediction = np.sum(power[selected]*psf[index, selected]); predicted[index] = prediction
            if prediction > 0 and power[index] <= prediction*10**(options["sidelobe_reject_db"]/10):
                rejected.append(index); continue
        selected.append(index)
        if len(selected) >= options["max_num_peaks"]:
            break
    return np.asarray(sorted(selected or [best])), np.asarray(sorted(rejected), dtype=int), predicted, psf


def rbd_decompose(signal, fs, theta, delays, bins=None, normalize_spectrum=False, multipath=False, multipath_options=None, psf=None):
    if signal.ndim != 2 or signal.shape[0] < 1 or signal.shape[1] < 2:
        raise ValueError("RBD signal must have shape [element,time] with at least two samples")
    theta = np.asarray(theta, dtype=float)
    if delays.shape != (signal.shape[0], len(theta)):
        raise ValueError("Steering delay matrix does not match array/angle dimensions")
    frequencies = np.arange(signal.shape[1]//2+1)*fs/signal.shape[1]
    bins = np.arange(len(frequencies)) if bins is None else np.asarray(bins, dtype=np.int64)
    frequencies = frequencies[bins]; omega = 2*np.pi*frequencies
    spectrum = np.fft.fft(signal, axis=1)[:, bins]
    normalized = spectrum/(np.linalg.norm(spectrum, axis=0, keepdims=True)+EPS)
    scale = float(np.sqrt(np.sum(np.abs(spectrum)**2)/signal.shape[0])) if normalize_spectrum else 1.0
    working = spectrum/(scale+EPS) if normalize_spectrum else spectrum
    beam = bartlett_beamformer(working, omega, delays)
    power = np.sum(np.abs(beam)**2, axis=0); power[~np.isfinite(power)] = -np.inf
    best = int(np.argmax(power))
    if multipath:
        defaults = {"peak_threshold_db": -6, "min_separation_deg": 2,
                    "max_num_peaks": len(theta), "sidelobe_reject_db": 3}
        selected, rejected, prediction, psf = select_beam_peaks(power, theta, delays, omega, defaults | (multipath_options or {}), psf)
    else:
        selected, rejected, prediction = np.asarray([best]), np.asarray([], dtype=int), np.zeros(len(theta))
    selected_power = np.where(np.isfinite(power[selected]) & (power[selected] >= 0), power[selected], 0.0)
    weights = selected_power/np.sum(selected_power) if np.sum(selected_power) > 0 else np.full(len(selected), 1/len(selected))
    rotations = np.exp(-1j*np.angle(beam[:, selected])).T
    components = normalized[:, :, None]*rotations.T[None, :, :]
    green = np.sum(components*weights[None, None, :], axis=2)
    return working, frequencies, {"green_freq": green, "theta_best": theta[best], "best_angle_idx": best,
             "selected_angle_idx": selected, "theta_selected": theta[selected], "selected_beam_power": power[selected],
             "num_selected_angles": len(selected), "beam_power": power, "signal_freq_scale": scale,
             "green_freq_weights": weights, "phase_rotation_components": rotations,
             "green_freq_components": components, "sidelobe_rejected_angle_idx": rejected,
             "sidelobe_prediction_power": prediction, "sidelobe_psf_power": psf}
