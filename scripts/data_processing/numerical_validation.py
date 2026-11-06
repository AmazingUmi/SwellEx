"""Independent synthetic numerical contracts for the pure Python generators."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import tempfile
import traceback

import h5py
import numpy as np
from scipy.io import savemat

from scripts.data_processing.dataset_validation import run_datasets_command
from scripts.data_processing.feature_math import (
    EPS, pair_indices, scm_feature, rbd_decompose, steering_delays,
    extend_profile, select_beam_peaks, sidelobe_psf, segments, combine_snapshots,
    build_split, select_frequency_bins,
)
from scripts.data_processing.generation_config import (
    ROOT, normalize_config, output_path, validate_generation_tables,
)
from scripts.data_processing.generation_core import (
    GENERATOR_VERSION, create_h5, load_channel, source_manifest, write_json,
)
from scripts.data_processing.input_contract import METHODS

TOLERANCE = 1e-12


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=TOLERANCE):
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, err_msg=message)


def raises(callback):
    try:
        callback()
    except (ValueError, TypeError):
        return
    raise AssertionError("Expected invalid input to be rejected")


def test_scm():
    fs, n, f = 64, 64, 8
    tone = np.exp(2j*np.pi*f*np.arange(n)/fs)
    a = np.asarray([1, 2*np.exp(0.3j), 0.5*np.exp(-0.7j)])
    b = np.asarray([2j, -0.5, 1])
    signal = np.concatenate([a[:, None]*tone, b[:, None]*tone], axis=1)
    i, j = pair_indices(3, diagonal=True)
    check(i.tolist() == [0, 0, 1, 0, 1, 2] and j.tolist() == [0, 1, 1, 2, 2, 2], "SCM pairs must follow MATLAB column-major upper triangle including diagonal")
    feature, info = scm_feature(signal, n, n, np.asarray([f]), i, j)
    expected = (np.outer(a, a.conj())/np.vdot(a, a).real + np.outer(b, b.conj())/np.vdot(b, b).real)/2
    close(feature[:, 0], expected[i, j], "SCM must average individually normalized snapshot outer products")
    covariance = np.zeros((3, 3), dtype=complex); covariance[i, j] = feature[:, 0]; covariance[j, i] = feature[:, 0].conj()
    close(covariance, covariance.conj().T, "SCM must be Hermitian")
    close(np.trace(covariance), 1.0, "Nonzero SCM must have unit trace")
    check(np.min(np.linalg.eigvalsh(covariance)) >= -TOLERANCE and info["num_snapshots"] == 2, "SCM must be PSD and use both snapshots")
    zero, _ = scm_feature(np.zeros((3, n)), n, n, np.asarray([f]), i, j)
    check(np.all(np.isfinite(zero)) and np.all(zero == 0), "SCM zero signal must remain finite zero")
    dc = np.asarray([100.0, -250.0, 75.0])[:, None]
    centered_hann, _ = scm_feature(
        signal + np.concatenate([dc * np.ones((1, n)), -2.0 * dc * np.ones((1, n))], axis=1),
        n, n, np.asarray([0, f]), i, j, norm_floor=1e-9, fft_window="hann", center_snapshots=True,
    )
    hann_window = np.hanning(n)[None, :]
    manual_a = np.fft.fft((a[:, None] * tone) * hann_window, axis=1)[:, [0, f]]
    manual_b = np.fft.fft((b[:, None] * tone) * hann_window, axis=1)[:, [0, f]]
    manual_a /= np.maximum(np.linalg.norm(manual_a, axis=0, keepdims=True), 1e-9)
    manual_b /= np.maximum(np.linalg.norm(manual_b, axis=0, keepdims=True), 1e-9)
    expected_hann = 0.5 * (manual_a[i] * manual_a[j].conj() + manual_b[i] * manual_b[j].conj())
    close(centered_hann, expected_hann, "Centered Hann SCM must match manual per-snapshot centered Hann outer products")
    raises(lambda: scm_feature(signal, n, n, np.asarray([f]), i, j, norm_floor=0.0))
    raises(lambda: scm_feature(signal, n, n, np.asarray([f]), i, j, norm_floor=-1e-12))
    raises(lambda: scm_feature(signal, n, n, np.asarray([f]), i, j, fft_window="hamming"))


def test_rbd():
    fs, n, f = 256, 256, 16
    theta = np.deg2rad(np.arange(-60, 61)); depths = np.arange(8)*2
    tau = steering_delays(theta, depths, np.asarray([1500, 1500]), np.asarray([0, 20]), True)
    index = int(np.flatnonzero(np.isclose(theta, np.deg2rad(25)))[0])
    wave = np.exp(0.37j)*np.exp(2j*np.pi*f*(tau[:, index, None]+np.arange(n)[None, :]/fs))
    spectrum, frequencies, result = rbd_decompose(wave, fs, theta, tau, np.asarray([f]))
    check(frequencies[0] == f, "RBD selected physical frequency must be 16 Hz")
    close(result["theta_best"], np.deg2rad(25), "Known wave must peak at 25 degrees")
    close(np.abs(spectrum), n, "First RBD output must preserve FFT amplitude")
    expected = np.exp(2j*np.pi*f*tau[:, index])/np.sqrt(len(depths))
    close(result["green_freq"][:, 0], expected, "Actual Green estimate must preserve steering phase and unit norm")
    close(np.sum(np.abs(result["green_freq"])**2, axis=0), 1, "Best-beam Green vectors must have unit per-frequency power")
    check(not np.allclose(np.sum(np.abs(spectrum)**2, axis=0), 1), "FFT negative control must differ from Green features")
    scaled, _, normalized = rbd_decompose(wave, fs, theta, tau, np.asarray([f]), normalize_spectrum=True)
    close(np.abs(scaled), 1, "Global spectrum normalization must preserve unit tone amplitude")
    close(normalized["green_freq"], result["green_freq"], "Green estimate must not depend on global spectrum scaling")
    _, full_frequencies, full = rbd_decompose(wave, fs, theta, tau)
    check(len(full_frequencies) == n//2+1 and np.all(np.isfinite(full["green_freq"])), "Full-axis decomposition must be finite")
    close(full["theta_best"], result["theta_best"], "Full and selected estimation must agree on arrival")
    close(full["green_freq"][:, f], expected, "Full FFT axis must retain same known Green bin")
    _, _, multipath = rbd_decompose(wave, fs, theta, tau, np.asarray([f]), multipath=True)
    check(np.any(np.isclose(multipath["theta_selected"], np.deg2rad(25))) and np.all(np.isfinite(multipath["green_freq"])), "Multipath selection must retain true arrival")
    check(np.all(np.linalg.norm(multipath["green_freq"], axis=0) <= 1+TOLERANCE), "Weighted Green combination cannot exceed unit norm")
    close(np.sum(multipath["green_freq_weights"]), 1, "Multipath power weights must sum to one")
    two_angles = np.deg2rad(np.asarray([-30,0,30]))
    two_delays = steering_delays(two_angles,np.asarray([0,1500/(2*f)]),
                                np.asarray([1500,1500]),np.asarray([0,100]),True)
    a = np.asarray([1,-1j]); b = np.asarray([1,1j]); array_vector = 3*a+2j*b
    mixed_wave = array_vector[:,None]*np.exp(2j*np.pi*f*np.arange(n)[None,:]/fs)
    _,_,two_beams = rbd_decompose(mixed_wave,fs,two_angles,two_delays,np.asarray([f]),multipath=True)
    check(two_beams["selected_angle_idx"].tolist()==[0,2], "Orthogonal unequal-power sources must both survive multipath selection")
    close(two_beams["green_freq_weights"],[9/13,4/13], "Multipath weights must use beam power, not equal or square-root-power weights")
    expected_mixed_green = array_vector/np.sqrt(26)*(9/13-4j/13)
    close(two_beams["green_freq"][:,0],expected_mixed_green, "Weighted phase-rotated Green estimate must match independent two-source closed form")
    biased_wave = wave + (np.arange(1, len(depths) + 1, dtype=float) * 50.0)[:, None]
    hann_spec, _, hann_rbd = rbd_decompose(
        biased_wave, fs, theta, tau, np.asarray([f]), fft_window="hann", center_window=True,
    )
    expected_hann_spec = np.fft.fft(
        (biased_wave - np.mean(biased_wave, axis=1, keepdims=True)) * np.hanning(n)[None, :],
        axis=1,
    )[:, [f]]
    close(hann_spec, expected_hann_spec, "Centered Hann RBD spectrum must match manual centered Hann FFT")
    close(hann_rbd["theta_best"], np.deg2rad(25), "Centered Hann RBD must recover true plane-wave arrival")
    close(np.sum(np.abs(hann_rbd["green_freq"])**2, axis=0), 1, "Centered Hann RBD Green vectors must keep unit norm")
    raises(lambda: rbd_decompose(wave, fs, theta, tau, np.asarray([f]), fft_window="blackman"))


def test_ray_delays_and_multipath():
    theta = np.deg2rad(np.asarray([-30, 0, 30])); depths = np.asarray([0, 10, 20])
    z = np.arange(0, 20.5, 0.5); c = np.full(len(z), 1500.0)
    ray = steering_delays(theta, depths, c, z, False)
    expected = depths[:, None]*np.sin(theta)[None, :]/1500
    close(ray, expected, "Constant sound speed ray integral must match plane-wave closed form")
    z = np.asarray([0, 5, 10, 15, 20], dtype=float); c = np.asarray([1510, 1504, 1500, 1503, 1512], dtype=float)
    theta = np.deg2rad(np.asarray([-30, 30])); depths = np.asarray([0, 20], dtype=float)
    ray = steering_delays(theta, depths, c, z, False)
    p = np.cos(np.deg2rad(30))**2/1500**2
    integrand = np.sqrt(1/c**2-p)
    above = 5*(integrand[0]/2+integrand[1]+integrand[2]/2)
    below = 5*(integrand[2]/2+integrand[3]+integrand[4]/2)
    close(ray, [[above, -above], [-below, below]], "Variable sound-speed ray delays must match independent quadrature")
    fallback_depths = np.asarray([5,20],dtype=float)
    fallback_angle = np.deg2rad(5)
    fallback = steering_delays(np.asarray([fallback_angle]), fallback_depths, c, z, False)
    fallback_expected = (fallback_depths-10)*np.sin(fallback_angle)/1500
    close(fallback[:,0], fallback_expected, "Turning-invalid later elements must trigger nonzero whole-column plane fallback")
    extended_z, extended_c = extend_profile(np.asarray([0, 1]), np.asarray([1500, 1490]), 2)
    close(extended_z, [0, 1, 1.5, 2], "CTD extension must use 0.5 m spacing")
    close(extended_c, [1500, 1490, 1490, 1490], "CTD bottom extension must be constant sound speed")
    angles = np.deg2rad(np.asarray([-40, -20, 0, 20, 40])); delays = np.asarray([0, 0.005])[:, None]*np.sin(angles)
    power = np.asarray([0, 10, 0, 8, 0], dtype=float); omega = np.asarray([2*np.pi*80])
    options = {"peak_threshold_db": -6, "min_separation_deg": 2, "max_num_peaks": 5, "sidelobe_reject_db": 3}
    independent_psf = np.eye(5)
    selected, rejected, _, _ = select_beam_peaks(power, angles, delays, omega, options, independent_psf)
    check(selected.tolist() == [1, 3] and rejected.size == 0, "Separated peaks without sidelobe leakage must both be retained")
    leakage_psf = np.eye(5); leakage_psf[3, 1] = 0.5
    selected, rejected, prediction, _ = select_beam_peaks(power, angles, delays, omega, options, leakage_psf)
    check(selected.tolist() == [1] and rejected.tolist() == [3] and prediction[3] == 5, "Predicted sidelobe leakage must reject weaker candidate")
    actual_psf = sidelobe_psf(delays, omega)
    close(np.diag(actual_psf), 1, "Beam point-spread response must have unit self-response")
    check(np.all(actual_psf >= 0) and np.all(actual_psf <= 1+TOLERANCE), "Point-spread powers must be bounded [0,1]")
    delay_difference = 0.013
    two_frequency_omega = 2*np.pi*np.asarray([16,40])
    closed_psf = sidelobe_psf(np.asarray([[0,0],[0,delay_difference]],dtype=float),two_frequency_omega)
    off_diagonal = np.mean(np.cos(two_frequency_omega*delay_difference/2)**2)
    close(closed_psf,[[1,off_diagonal],[off_diagonal,1]], "PSF must incoherently sum squared response magnitudes across frequencies")


def test_segments_and_splits():
    config = normalize_config({"dataset": {"segment_start_s": 0, "segment_end_s": 9}})["dataset"]
    base = segments(np.zeros((2, 640)), 64, 1, 1, 0, 9, np.asarray([0, 10]), np.asarray([1, 3]))
    close(base["sample_start"], np.arange(0, 640, 64), "Segment sample starts must align to exact grid")
    close(base["sample_stop"], np.arange(64, 641, 64), "Segment stop samples must be exclusive Python indices")
    splits, _ = build_split(base, config)
    check(splits["train"].tolist() == [0,1,2,3,5,6,7,8] and splits["test"].tolist() == [4,9], "Full interleaved split must be deterministic 4:1")
    combined = combine_snapshots(base, 4, 3, np.asarray([0,10]), np.asarray([1,3]))
    close(combined["start"], np.arange(7), "Ns4 overlap3 must advance by one snapshot")
    close(combined["stop"], np.arange(4,11), "Combined windows must span four snapshots")
    check(combined["duration_s"] == 4 and combined["step_s"] == 1, "Combined window metadata must match Ns/overlap")
    near = base | {"start": np.arange(11), "center": np.arange(11)+0.5,
                   "labels": np.abs(np.arange(11)-5), "valid": np.ones(11, dtype=bool)}
    options = config | {"split_strategy": "range_overlap", "half_duration_s": 4, "gap_s": 1, "train_side": "before"}
    before, meta = build_split(near, options); after, _ = build_split(near, options | {"train_side": "after"})
    check(before["train"].tolist() == after["test"].tolist() and before["test"].tolist() == after["train"].tolist(), "Range-overlap train-side reversal must swap the selected sides")
    check(np.all(before["train"] < meta["range_min_segment_idx"]-1) and np.all(before["test"] > meta["range_min_segment_idx"]-1), "Range-overlap windows must be on opposite sides of closest approach")
    asymmetric = near | {"labels": np.asarray([10,8,6,4,2,0,1,2,3,4,5], dtype=float)}
    options = options | {"gap_s": 0}
    selected, metadata = build_split(asymmetric, options)
    check(selected["train"].tolist() == [3,4] and selected["test"].tolist() == [7,8,9], "Overlap [2,4] must exclude out-of-range windows and allow unequal side counts")
    check(metadata["range_overlap_min_km"] == 2 and metadata["range_overlap_max_km"] == 4, "Metadata must describe the candidate-side range intersection")
    reversed_split, _ = build_split(asymmetric, options | {"train_side": "after", "test_step_s": 2})
    check(reversed_split["train"].tolist() == [7,8,9] and reversed_split["test"].tolist() == [3], "Test stride must apply after range filtering and train-side selection")
    invalid = asymmetric | {"valid": np.arange(11) != 3}
    filtered, _ = build_split(invalid, options)
    check(filtered["train"].tolist() == [4] and filtered["test"].tolist() == [7,8,9], "An invalid window must not remove an unrelated window on the other side")
    truncated = near | {"start": np.arange(7), "center": np.arange(7)+0.5,
                        "labels": np.asarray([4,2,0,1,2,3,4]), "valid": np.ones(7, dtype=bool)}
    boundary_split, _ = build_split(truncated, options)
    check(boundary_split["train"].tolist() == [0,1] and boundary_split["test"].tolist() == [4,5,6], "Recording boundaries must clip each candidate side independently")
    raises(lambda: build_split(near | {"labels": np.asarray([10,9,8,7,6,0,1,2,3,4,5])}, options))
    raises(lambda: build_split(near | {"labels": np.asarray([20,18,16,2,1,0,2,3,17,19,21])}, options | {"half_duration_s": 2}))
    raises(lambda: build_split(near | {"labels": np.asarray([20,18,16,1,4,0,2,3,17,19,21])}, options | {"half_duration_s": 2}))
    raises(lambda: build_split(near, options | {"split_strategy": "Range_nearby"}))


def test_frequency_training_only():
    fs, n = 64, 64; t = np.arange(n)/fs
    signal = np.asarray([np.concatenate([np.exp(2j*np.pi*8*t), 100*np.exp(2j*np.pi*20*t)])])
    windows = segments(signal, fs, 1, 1, 0, 1, np.asarray([0,2]), np.asarray([1,2]))
    options = {"adapt_num_bins": 1}
    bins, frequencies, info = select_frequency_bins(np.arange(33), ["adapt"], options, signal=signal, windows=windows, train_indices=np.asarray([0]))
    check(bins.tolist() == [8] and frequencies.tolist() == [8] and info["adapt_training_segment_idx"].tolist() == [1], "Adaptive bins must use only explicit training windows")
    signal[:, n:] = 1e6*np.exp(2j*np.pi*25*t)
    bins2, frequencies2, _ = select_frequency_bins(np.arange(33), ["adapt"], options, signal=signal, windows=windows, train_indices=np.asarray([0]))
    check(np.array_equal(bins,bins2) and np.array_equal(frequencies,frequencies2), "Changing holdout spectrum must not change adaptive bins")
    bins, frequencies, _ = select_frequency_bins(np.arange(751), ["deep","shallow"])
    check(len(bins) == 22 and np.all(np.diff(bins)>0), "Named frequency union must have 22 sorted unique bins")
    close(frequencies, sorted([49,64,79,94,112,130,148,166,201,235,283,338,388,109,127,145,163,198,232,280,335,385]), "Named physical frequencies must be preserved")
    raises(lambda: select_frequency_bins(np.arange(33), ["full","adapt"]))
    raises(lambda: select_frequency_bins(np.arange(33), ["adapt"], options, signal=signal, windows=windows, train_indices=np.asarray([],dtype=int)))
    dc_signal = np.asarray([np.concatenate([1e5 + np.exp(2j*np.pi*8*t), np.zeros(n)])])
    uncentered_bins, _, _ = select_frequency_bins(
        np.arange(33), ["adapt"], options, signal=dc_signal, windows=windows,
        train_indices=np.asarray([0]), fft_window="hann", center_snapshots=False,
    )
    centered_bins, _, _ = select_frequency_bins(
        np.arange(33), ["adapt"], options, signal=dc_signal, windows=windows,
        train_indices=np.asarray([0]), fft_window="hann", center_snapshots=True,
    )
    check(uncentered_bins.tolist() == [1] and centered_bins.tolist() == [8],
          "Adaptive Hann selection must apply snapshot centering before windowing")


def test_configuration():
    defaults = normalize_config({})
    check(defaults["dataset"]["segment_end_s"] is None and defaults["dataset"]["train_side"] == "before", "Canonical manual defaults must retain full record and shared explicit train side")
    check(defaults["dataset"]["ctd_file"] == "i9605.prn"
          and defaults["dataset"]["signal_centering"] == "full_record"
          and defaults["dataset"]["fft_window"] == "boxcar",
          "Dataset defaults must expose ctd_file, full_record centering, and boxcar FFT window")
    check(defaults["rbd"]["angle_min_deg"] == -90.0
          and defaults["rbd"]["angle_max_deg"] == 90.0
          and defaults["rbd"]["num_angles"] == 181,
          "RBD defaults must expose canonical angle grid")
    check(defaults["scm"] == {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0, "norm_floor": 1e-12},
          "SCM defaults must hold snapshot and norm_floor parameters")
    d_tab, r_tab, s_tab = validate_generation_tables({}, {}, {})
    check((d_tab, r_tab, s_tab) == (defaults["dataset"], defaults["rbd"], defaults["scm"]),
          "validate_generation_tables must match normalize_config table defaults")
    sentinels = normalize_config({"dataset": {"segment_start_s": -1, "segment_end_s": -1.0, "test_step_s": 0}})["dataset"]
    check(sentinels["segment_start_s"] is None and sentinels["segment_end_s"] is None and sentinels["test_step_s"] is None,
          "TOML/JSON sentinels -1 and 0 must normalize to None")
    check(normalize_config({"dataset": {"test_step_s": 0.0}})["dataset"]["test_step_s"] is None,
          "Float 0.0 test_step_s sentinel must normalize to None")
    custom = normalize_config({
        "dataset": {
            "split_strategy": "range_overlap",
            "ctd_file": "i9605_alt.prn",
            "signal_centering": "per_window",
            "fft_window": "hann",
            "frequency_selection_config": {"adapt_max_num_segments": 0},
            "test_step_s": 3.0,
        },
        "rbd": {"angle_min_deg": -45.0, "angle_max_deg": 45.0, "num_angles": 91},
        "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 1, "norm_floor": 1e-8},
    })
    check(custom["dataset"]["fft_window"] == "hann"
          and custom["dataset"]["signal_centering"] == "per_window"
          and custom["scm"]["num_snapshots_per_segment"] == 4
          and custom["rbd"]["num_angles"] == 91,
          "Valid custom dataset/rbd/scm settings must normalize cleanly")
    scm_full = normalize_config({"dataset": {"frequency_selection_modes": ["full"]}}, methods=("SCM",))
    check(scm_full["dataset"]["frequency_selection_modes"] == ["full"],
          "SCM-only normalization must allow full frequency selection without RBD full estimation")
    rbd_step = normalize_config(
        {"dataset": {"test_step_s": 1.0}, "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 2}},
        methods=("RBD",),
    )
    check(rbd_step["dataset"]["test_step_s"] == 1.0,
          "RBD-only normalization must ignore SCM combined step and SCM overlap on full_interleaved")
    invalid_cases = [{"typo": 1}, {"dataset": {"fs": float("nan")}},
                     {"dataset": {"num_snapshots_per_segment": 1}},
                     {"dataset": {"snapshot_overlap_count": 0}},
                     {"scm": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 1}},
                     {"scm": {"num_snapshots_per_segment": 4.0}},
                     {"scm": {"snapshot_overlap_count": 0.0}},
                     {"scm": {"norm_floor": 0.0}},
                     {"scm": {"norm_floor": -1e-12}},
                     {"scm": {"norm_floor": 1.5}},
                     {"scm": {"unknown_scm_field": 1}},
                     {"rbd": {"normalize_spectrum": 1}},
                     {"rbd": {"angle_min_deg": -95.0}},
                     {"rbd": {"angle_max_deg": 95.0}},
                     {"rbd": {"angle_min_deg": 10.0, "angle_max_deg": 10.0}},
                     {"rbd": {"num_angles": 1}},
                     {"rbd": {"num_angles": 90.0}},
                     {"result_path": "../report.json"},
                     {"dataset": {"frequency_selection_modes": ["full","deep"]}},
                     {"dataset": {"frequency_selection_modes": ["full"]}},
                     {"dataset": {"ctd_file": ""}},
                     {"dataset": {"ctd_file": "../i9605.prn"}},
                     {"dataset": {"ctd_file": "bad..name.prn"}},
                     {"dataset": {"ctd_file": ".hidden"}},
                     {"dataset": {"signal_centering": "window"}},
                     {"dataset": {"fft_window": "hamming"}},
                     {"dataset": {"segment_start_s": -2}},
                     {"dataset": {"segment_end_s": -0.5}},
                     {"dataset": {"test_step_s": -1}},
                     {"dataset": {"segment_start_s": False}},
                     {"dataset": {"test_step_s": False}},
                     {"dataset": {"test_step_s": 0.5}},
                     {"dataset": {"split_strategy": "range_overlap", "test_step_s": 2.0}, "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 1}},
                     {"dataset": {"split_strategy": "full_interleaved", "segment_step_s": 0.5}},
                     {"dataset": {"split_strategy": "full_interleaved"}, "scm": {"num_snapshots_per_segment": 2, "snapshot_overlap_count": 1}},
                     {"dataset": {"split_strategy": "range_overlap", "gap_s": 0.0}, "scm": {"num_snapshots_per_segment": 4, "snapshot_overlap_count": 3}},
                     {"schema_version": 1.0}, {"schema_version": True},
                     {"dataset": {"fs": 1500.0}},
                     {"dataset": {"train_test_ratio": [4.0,1]}},
                     {"dataset": {"split_strategy": "periodic"}},
                     {"dataset": {"split_strategy": "Range_nearby"}},
                     {"dataset": {"frequency_selection_config": {"adapt_max_num_segments": 2.0}}},
                     {"dataset": {"frequency_selection_config": {"adapt_max_num_segments": -1}}},
                     {"dataset": {"frequency_selection_config": {"mel_num_bins": 64.0}}},
                     {"rbd": {"multipath_options": {"max_num_peaks": 2.0}}}]
    for invalid in invalid_cases:
        raises(lambda invalid=invalid: normalize_config(invalid))
    raises(lambda: normalize_config({"dataset": {"test_step_s": 0.5}}, methods=("RBD",)))
    raises(lambda: normalize_config({"dataset": {"frequency_selection_modes": ["full"]}}, methods=("RBD",)))
    raises(lambda: normalize_config({}, methods=()))
    raises(lambda: normalize_config({}, methods=("UNKNOWN",)))


def test_numeric_mat_and_hdf_contract(report_directory):
    with tempfile.TemporaryDirectory(prefix="generation_numeric_", dir=report_directory) as temporary:
        root = Path(temporary); channel_path = root/"channel.mat"
        savemat(channel_path, {"x": np.arange(64, dtype=float).reshape(-1, 1)})
        close(load_channel(channel_path), np.arange(64), "Numeric MAT x input must be read through SciPy without orientation changes")
        savemat(channel_path, {"x": np.asarray(["not numeric"],dtype=object)})
        raises(lambda: load_channel(channel_path))
        inputs = {"depths": np.asarray([10,20,30]), "range_times": np.asarray([0,5]), "ranges": np.asarray([1,2])}
        windows = segments(np.zeros((3,256)),64,1,1,0,3,inputs["range_times"],inputs["ranges"])
        i,j = pair_indices(3,True)
        path=root/"scm.h5"
        with create_h5(path,"SCM",np.asarray([0,2]),windows,inputs,np.asarray([8.0]),len(i),i,j) as handle:
            handle["X"][:] = 0
        with h5py.File(path,"r") as handle:
            check(handle["X"].shape == (2,6,1,2), "Generated HDF X must use native logical axes")
            check(handle["X"].attrs["axis_order"] == "sample,pair,frequency,real_imag" and handle["X"].attrs["storage_axis_order"] == "sample,pair,frequency,real_imag", "HDF axis declarations must identify native storage")
            close(handle["time/sample_start_idx"][:,0],[1,129],"Persisted sample starts must be 1 based")
            close(handle["time/sample_stop_idx"][:,0],[64,192],"Persisted sample stops must be inclusive MATLAB indices")
            close(handle["split/source_segment_idx"][:,0],[1,3],"Persisted source window indices must be 1 based")
            close(handle["pair/numerator_element_idx"][:,0],i+1,"Persisted pairs must retain MATLAB column-major ordering")


def run_validation(report_path):
    report_path = output_path(str(report_path)); report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version":1,"status":"PASS","purpose":"engineering_numerical_validation",
              "generator_version":GENERATOR_VERSION,"python_version":platform.python_version(),
              "tolerance":TOLERANCE,"source_manifest":source_manifest(),"checks":[]}
    tests = [test_scm, test_rbd, test_ray_delays_and_multipath,
             test_segments_and_splits,
             test_frequency_training_only,
             test_configuration,
             lambda: test_numeric_mat_and_hdf_contract(report_path.parent)]
    names = ["scm_hermitian_psd_trace_pairs_zero","rbd_wave_phase_normalization",
             "rbd_ray_integral_multipath_sidelobes","segmentation_split_snapshots",
             "frequency_training_only","strict_configuration","numeric_mat_hdf_contract"]
    for name, callback in zip(names,tests):
        check_result = {"name":name,"status":"PASS"}
        try:
            callback()
        except Exception as failure:
            check_result.update(status="FAIL",error_type=type(failure).__name__,detail=str(failure),traceback=traceback.format_exc())
            report["status"]="FAIL"
        report["checks"].append(check_result)
        print(f"{check_result['status']} {name}",flush=True)
    write_json(report_path,report)
    return report


def run_cli(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "datasets":
        parser = argparse.ArgumentParser(prog="validate.py datasets", allow_abbrev=False,
                                         description="Validate generated HDF5 datasets and cross-route equivalence.")
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--report", type=Path, required=True)
        parser.add_argument("--expected-config", type=Path)
        parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
        args = parser.parse_args(raw_argv[1:])
        return run_datasets_command(root=args.root, report_path=args.report,
                                    expected_config=args.expected_config, methods=args.methods)
    parser = argparse.ArgumentParser(
        description="Run pure-Python generator numerical validation, or pass `datasets --root ... --report ...`.",
        allow_abbrev=False,
    )
    parser.add_argument("--report", type=Path, default=ROOT/"outputs/validation/data_processing_validation.json")
    args = parser.parse_args(raw_argv)
    report = run_validation(args.report)
    print(f"Numerical validation {report['status']}: {args.report}")
    return 0 if report["status"] == "PASS" else 1
