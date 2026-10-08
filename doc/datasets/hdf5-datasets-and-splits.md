# HDF5 Datasets and Split Strategies

## Generation

The canonical Python entry is `scripts/data_processing/generate.py --config
<JSON>`. Data/feature math is in `generation_core.py` and `feature_math.py`;
strict configuration is defined in `generation_config.py`. The
[haiqin workflow](../workflows/remote-validation.md) freezes
configuration and performs generation remotely with uv.

Inputs are numeric VLA channel MAT files, range/position text files, and raw
`Origindata/environments/ctds/i9605.prn` CTD depth/sound-speed columns. SciPy
reads numeric MAT files; h5py reads numeric v7.3 datasets. Object/struct/reference
channel data are rejected. Before windowing and splitting, each element's mean
over the entire loaded recording is subtracted; this preprocessing includes
held-out times, unlike adaptive frequency fitting on training windows only.
All outputs live under `outputs/Datasets/`.

`generation_result.json` reports PASS/FAIL and route-to-directory mappings.
Each dataset has side JSON metadata recording configuration, split, frequency
selection, time geometry and source identities. Existing MATLAB implementations
are static references under `scripts/data_processing/matlab_reference/`.

## Stored Contracts

Python writes `/X` in its logical axis order and declares `axis_order` plus
`storage_axis_order`. Loaders also recognize earlier reversed MATLAB storage.
`/X` is float32; the vector fields below are stored as `[N,1]` columns.
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
               / (sum_s abs(FFT_j,s(f))^2 + denominator_floor + EPS)
```

The ELM floor is `1e-6` times the median finite FFT power across the current
combined window; `EPS` is float64 machine epsilon. SCM averages outer products
after normalizing each snapshot's array vector with a norm floor of `1e-12`;
its formula, trace qualification, and GRNN vectorization are defined in the
[SCM/GRNN method note](../methods/scm-grnn.md). Synthetic tests check known
ratios, covariance properties, pair sequence, and zero-power behavior.

RBD retains variable-sound-speed steering-delay integration, plane-wave
fallback and Bartlett beamforming. `rbd.beam_selection` is `best` or `multipath`;
accepted multipath components are phase rotated and beam-power weighted.
`rbd.frequency_estimation` is `selected` or `full`. Standalone generation with
`full` overrides RBD's axis to all one-sided FFT bins while the other routes
may retain their selected modes. The remote comparison workflow requires
shared modes `["full"]` in this case so all routes align. The stored feature is
the actual Green estimate. Earlier segmentation source saved the FFT return
value under that name; regenerate those RBD datasets before comparisons.
RBD normalizes each spectrum vector by its norm plus float64 machine epsilon;
norms well above epsilon therefore approach one. With an expected generation
configuration,
the validator requires best-beam columns above `1e-8` to be within `2e-6` of
unit norm, exempting DC in full mode; multipath columns must not exceed
`1+2e-6`. These checks distinguish Green features from raw FFT amplitudes,
without certifying the physical estimator.

RBD diagnostics are stored under `/rbd`: theta grid/best/selected angles,
accepted angle count/powers, full beam powers and spectrum scale. Unused arrival
slots are NaN; model input `/X` remains finite. Shapes and peak-selection
behavior are defined in [RBD multipath](../methods/rbd-multipath-detection.md).

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

For model operations, see [training and prediction](../workflows/training-and-prediction.md).
