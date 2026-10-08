# HDF5 Datasets and Split Strategies

## Generation

The canonical Python entry is `scripts/data_processing/generate.py --config
<JSON>`. Data/feature math is in `generation_core.py` and `feature_math.py`;
strict configuration is defined in `generation_config.py`. The
[haiqin workflow](../20_python_workflows/30_remote_validation.md) freezes
configuration and performs generation remotely with uv.

Inputs are numeric VLA channel MAT files, range/position text files, and raw
`Origindata/environments/ctds/i9605.prn` CTD depth/sound-speed columns. SciPy
reads numeric MAT files; h5py reads numeric v7.3 datasets. Object/struct/reference
channel data are rejected. All outputs live under `outputs/Datasets/`.

`generation_result.json` reports PASS/FAIL and route-to-directory mappings.
Each dataset has side JSON metadata recording configuration, split, frequency
selection, time geometry and source identities. Existing MATLAB implementations
are static references under `scripts/data_processing/matlab_reference/`.

## Stored Contracts

Python writes `/X` in its logical axis order and declares `axis_order` plus
`storage_axis_order`. Loaders also recognize earlier reversed MATLAB storage.
Shared fields:

```text
/y_range_km                  km labels at combined window centers
/valid_sample                valid-label mask
/time/window_start_s
/time/window_center_s
/time/window_stop_s
/time/sample_start_idx       1-based inclusive sample index
/time/sample_stop_idx        1-based inclusive sample index
/frequency/freq_hz
/array/depth_m
/array/element_idx           1-based array IDs
/range/range_time_s
/range/range_km_raw
/split/source_segment_idx    1-based source-window identity
```

Labels interpolate the original range/time curve; out-of-coverage windows are
invalid. Validation checks finite features, sample endpoints/times, raw interval
split separation, label interpolation, metadata agreement, and route alignment.

## Feature Routes

| Route | Logical X | Feature | Pair count |
|---|---|---|---|
| RBD | `[S,E,F,2]` | equivalent Green estimate | — |
| ELM | `[S,P,F,2]` | least-squares frequency ratios | `E(E-1)/2` |
| SCM | `[S,P,F,2]` | normalized spatial covariance | `E(E+1)/2` |

Last-axis channels store real then imaginary components. Network input is
`[batch,2,E|P,F]`. ELM/SCM pair order traverses columns first and stores the
1-based pair indices in `/pair/numerator_element_idx` and
`/pair/denominator_element_idx`. ELM requires `i<j`; SCM requires `i<=j`.

ELM uses:

```text
ratio(i,j,f) = sum_s FFT_i,s(f) * conj(FFT_j,s(f))
               / (sum_s abs(FFT_j,s(f))^2 + denominator_floor)
```

SCM normalizes each snapshot's array vector before outer-product averaging:

```text
C_f = mean_s ((x_s / ||x_s||) (x_s / ||x_s||)^H)
```

Zero-power inputs are handled explicitly. Synthetic tests check known ratios,
Hermitian/PSD covariance, trace, pair sequence and zero-power behavior.

RBD retains variable-sound-speed steering-delay integration, plane-wave
fallback and Bartlett beamforming. `rbd.beam_selection` is `best` or `multipath`;
accepted multipath components are phase rotated and beam-power weighted.
`rbd.frequency_estimation` is `selected` or `full`; full estimation uses shared
frequency modes `["full"]` for cross-route comparison. The stored feature is
the actual Green estimate. Earlier segmentation source saved the FFT return
value under that name; regenerate those RBD datasets before comparisons.
Best-beam Green vectors have unit norm for active frequencies; zero bins and
multipath mixtures have separate validation bounds.

RBD diagnostics are stored under `/rbd`: theta grid/best/selected angles,
accepted angle count/powers, full beam powers and spectrum scale. Unused arrival
slots are NaN; model input `/X` remains finite.

## Windows, Frequencies, and Splits

Dataset configuration controls `segment_duration_s`, `segment_step_s`,
`segment_start_s`, `segment_end_s`, `num_snapshots_per_segment` and
`snapshot_overlap_count`. ELM/SCM combine Ns consecutive base snapshots and
advance by `(Ns-overlap)` base windows. Their labels/times describe the combined
window. RBD operates on base windows. Remote comparison suites require Ns=1,
no overlap, and disjoint base windows so all routes evaluate identical rows.

Frequency modes are arrays of `full`, `mel`, `deep`, `shallow`, `adapt`.
`full` is exclusive. Default deep targets are
`[49,64,79,94,112,130,148,166,201,235,283,338,388]` Hz; shallow targets are
`[109,127,145,163,198,232,280,335,385]` Hz. Mel uses 64 bins by default and
adapt uses 16. Target frequencies map to nearest one-sided FFT bins; duplicates
are removed and bins are sorted. Adapt fits valid training windows only and
freezes one dataset-level axis for held-out testing. Explicit overrides are
allowed in standalone `dataset.frequency_selection_config`.

`periodic` uses deterministic interleaving by `train_test_ratio` (default 4:1).
`Range_nearby` uses the minimum-range location, `half_duration_s`, `gap_s`,
`train_side` (`before`/`after`) and optional test-side `test_step_s`. Dataset
names encode the route, estimator and window/frequency settings;
`manual_dataset_variant_tag` appends a safe suffix.

For training and prediction, see [training](../20_python_workflows/00_trainable_training.md)
and [prediction](../20_python_workflows/10_trainable_prediction.md).
