"""Archived ELM (element least-squares ratio) feature and its numerical check.

Removed from the active ``scripts/data_processing/common`` modules when the
project was reduced to RBD/SCM. Restore ``elm_feature`` into ``feature_math.py``
and ``test_elm`` into ``numerical_validation.py`` to reactivate it.
"""
from __future__ import annotations

import numpy as np

from scripts.data_processing.common.feature_math import EPS, pair_indices, snapshot_fft

# check/close are the helpers defined in numerical_validation.py.


def elm_feature(signal, samples, step, bins, pair_i, pair_j, denominator_floor_relative=1e-6):
    spectra = snapshot_fft(signal, samples, step, bins)
    power = np.abs(spectra)**2
    finite = power[np.isfinite(power)]
    floor = denominator_floor_relative * (np.median(finite) if finite.size else 0.0)
    numerator = np.sum(spectra[pair_i] * np.conj(spectra[pair_j]), axis=2)
    denominator = np.sum(np.abs(spectra[pair_j])**2, axis=2)+floor+EPS
    return numerator/denominator, {"denom_floor": float(floor), "num_snapshots": spectra.shape[2]}


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
