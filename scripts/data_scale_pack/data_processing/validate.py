#!/usr/bin/env python3
"""Independent synthetic numerical contracts for the pure Python generators."""
from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import h5py
import numpy as np
from scipy.io import savemat

from scripts.data_scale_pack.data_processing.feature_math import (
    EPS, pair_indices, scm_feature, elm_feature, rbd_decompose, steering_delays,
    extend_profile, select_beam_peaks, sidelobe_psf, segments, combine_snapshots,
    build_split, select_frequency_bins,
)
from scripts.data_scale_pack.data_processing.generation_config import ROOT, normalize_config, output_path
from scripts.data_scale_pack.data_processing.generation_core import (
    write_json, source_manifest, load_channel, create_h5, GENERATOR_VERSION,
)

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


def test_elm():
    fs, n, f = 64, 64, 8
    tone = np.exp(2j*np.pi*f*np.arange(n)/fs)
    gains = np.asarray([1, 2*np.exp(0.3j), 0.5*np.exp(-0.7j)])
    signal = np.concatenate([gains[:, None]*tone, 2*gains[:, None]*tone, 0.5*gains[:, None]*tone], axis=1)
    i, j = pair_indices(3)
    check(i.tolist() == [0, 0, 1] and j.tolist() == [1, 2, 2], "ELM pairs must follow strict column-major upper triangle")
    ratios, info = elm_feature(signal, n, n, np.asarray([f]), i, j, 0)
    close(ratios[:, 0], gains[i]/gains[j], "Known element ratios must be recovered by least squares")
    check(info["num_snapshots"] == 3, "ELM must combine three snapshots")
    # An independent weighted LS oracle distinguishes sum-of-products from
    # averaging per-snapshot ratios: numerator [2,8], denominator [1,2].
    weighted = np.vstack([np.concatenate([2*tone, 8*tone]), np.concatenate([tone, 2*tone])])
    ratio, _ = elm_feature(weighted, n, n, np.asarray([f]), np.asarray([0]), np.asarray([1]), 0)
    close(ratio[0, 0], (2*1+8*2)/(1**2+2**2), "ELM must estimate weighted LS, not mean snapshot ratios")
    regularized, info = elm_feature(weighted, n, n, np.asarray([f]), np.asarray([0]), np.asarray([1]), 0.5)
    # The four selected powers are n^2*[4,64,1,4], whose median is 4*n^2.
    # A relative floor of .5 adds 2*n^2 to denominator 5*n^2.
    close(info["denom_floor"], 2*n**2, "ELM relative floor must use median selected snapshot power")
    close(regularized[0,0], 18/7, "ELM regularization must add one median-relative floor after summing snapshots")
    signal[1:] = 0
    ratios, _ = elm_feature(signal, n, n, np.asarray([f]), i, j)
    check(np.all(np.isfinite(ratios)) and np.all(ratios == 0), "Zero denominator must produce finite zero")
    ratios, _ = elm_feature(np.zeros((3, n)), n, n, np.asarray([f]), i, j, 0)
    check(np.all(ratios == 0), "Completely zero ELM signal must be finite zero")


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
    # Two sensors spaced c/(2*f) make +/-30 degree steering vectors
    # a=[1,-i], b=[1,i] orthogonal. Sources 3*a and 2i*b yield powers 9:4;
    # their quadrature phases keep the intervening zero-degree beam lower.
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
    # Direct trapezoid quadrature written as endpoint-weighted scalar sums.
    integrand = np.sqrt(1/c**2-p)
    above = 5*(integrand[0]/2+integrand[1]+integrand[2]/2)
    below = 5*(integrand[2]/2+integrand[3]+integrand[4]/2)
    close(ray, [[above, -above], [-below, below]], "Variable sound-speed ray delays must match independent quadrature")
    # The 5 m element has a valid 5--10 m integral at 5 degrees, while
    # the 20 m element includes c=1512 where the ray integrand is negative.
    # The fallback must overwrite the already computed first element too.
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


def test_configuration():
    defaults = normalize_config({})
    check(defaults["dataset"]["segment_end_s"] is None and defaults["dataset"]["train_side"] == "before", "Canonical manual defaults must retain full record and shared explicit train side")
    for invalid in ({"typo": 1}, {"dataset": {"fs": float("nan")}},
                    {"dataset": {"num_snapshots_per_segment": 1,"snapshot_overlap_count": 1}},
                    {"rbd": {"normalize_spectrum": 1}}, {"result_path": "../report.json"},
                    {"dataset": {"frequency_selection_modes": ["full","deep"]}},
                    {"schema_version": 1.0}, {"schema_version": True},
                    {"dataset": {"fs": 1500.0}},
                    {"dataset": {"num_snapshots_per_segment": 4.0}},
                    {"dataset": {"snapshot_overlap_count": 0.0}},
                    {"dataset": {"train_test_ratio": [4.0,1]}},
                    {"dataset": {"split_strategy": "periodic"}},
                    {"dataset": {"split_strategy": "Range_nearby"}},
                    {"dataset": {"frequency_selection_config": {"adapt_max_num_segments": 2.0}}},
                    {"dataset": {"frequency_selection_config": {"mel_num_bins": 64.0}}},
                    {"rbd": {"multipath_options": {"max_num_peaks": 2.0}}}):
        raises(lambda invalid=invalid: normalize_config(invalid))


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
    tests = [test_scm,test_elm,test_rbd,test_ray_delays_and_multipath,test_segments_and_splits,
             test_frequency_training_only,test_configuration,
             lambda: test_numeric_mat_and_hdf_contract(report_path.parent)]
    names = ["scm_hermitian_psd_trace_pairs_zero","elm_ls_ratios_zero","rbd_wave_phase_normalization",
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


def main():
    parser=argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report",type=Path,default=ROOT/"outputs/validation/data_processing_validation.json")
    args=parser.parse_args()
    report=run_validation(args.report)
    print(f"Numerical validation {report['status']}: {args.report}")
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
