# HDF5 Datasets and Split Strategies

## Generation

The [shell workflow](../workflows/remote-validation.md) generates and validates
RBD/ELM/SCM datasets on haiqin. The canonical Python generator is
`scripts/base_pack/data_processing/generate.py`, with a required `--config PATH` JSON file.

| Source | Responsibility |
|---|---|
| `generation_config.py` | Strict configuration fields and validation |
| `generation_core.py` | Inputs, windows, splits, and HDF5 output |
| `feature_math.py` | Feature calculations |

Inputs are numeric VLA channel MAT files, range/position text files, and CTD
depth/sound-speed columns from `Origindata/environments/ctds/i9605.prn`.
SciPy reads numeric MAT files; h5py reads numeric v7.3 datasets.
Object/struct/reference channel data are rejected.

Before windowing and splitting, the generator subtracts each element's mean
over the entire loaded recording. This includes held-out times; adaptive
frequency fitting instead uses training windows only.

Outputs go to `outputs/Datasets/`. `generation_result.json` records PASS/FAIL
and route-to-directory mappings. Each dataset has JSON metadata for its
configuration, split, frequency selection, time geometry, and source identities.
Its `input_files` records each actual raw input's path, byte size, modification
time, and SHA256 for later reuse checks.
Dataset directory names end with a 12-character configuration fingerprint.
It hashes the normalized dataset/RBD settings, schema, and generator version;
project and report locations do not affect it. Different settings use different
directories. Generation refuses an existing directory, including an incomplete
one, instead of overwriting or mixing files. Select a new
`manual_dataset_variant_tag` when a separate regeneration is needed.

Each HDF5 file is written to a temporary file and published only after closing
successfully. Publication refuses an existing final file; a failed write leaves
no partial final file. This is a file-level guarantee, not an atomic transaction
across all three feature routes. MATLAB files under
`scripts/base_pack/data_processing/matlab_reference/` are static references.

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
/array/element_idx           ordered array positions 1..E
/range/range_time_s
/range/range_km_raw
/split/source_segment_idx    1-based source-window identity
```

Labels interpolate the original range/time curve; out-of-coverage windows are
invalid. Validation checks finite features, sample endpoints/times, raw interval
split separation, label interpolation, metadata agreement, and route alignment.

Loaders require complete physical-axis metadata, including for reversed MATLAB
storage. They reject missing fields and compare `input_signature` across files,
training/validation data, and saved artifacts. Equal tensor shapes with changed
frequencies, array geometry, or pair order are rejected.

| Required axis fields | Contract |
|---|---|
| `/frequency/freq_hz` | Finite, nonnegative, strictly increasing; matches F |
| `/array/element_idx`, `/array/depth_m` | Ordered positions `1..E`; finite, nonnegative, strictly increasing depths |
| ELM/SCM `/pair/numerator_element_idx`, `/pair/denominator_element_idx` | Complete 1-based column-major pair sequence for the route |

Loaders filter invalid/nonfinite labels and reject nonfinite loaded features.
The input signature describes axes; it is not a source-file content hash or
proof of independent train/test windows.

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

### ELM and SCM

ELM uses:

```text
ratio(i,j,f) = sum_s FFT_i,s(f) * conj(FFT_j,s(f))
               / (sum_s abs(FFT_j,s(f))^2 + denominator_floor + EPS)
```

The ELM floor is `1e-6` times the median finite FFT power across the current
combined window; `EPS` is float64 machine epsilon. SCM averages outer products
after normalizing each snapshot's array vector with a norm floor of `1e-12`;
its formula, trace qualification, and GRNN vectorization are defined in the
[GRNN method note](../methods/grnn.md). Synthetic tests check known
ratios, covariance properties, pair sequence, and zero-power behavior.

### RBD

RBD uses variable-sound-speed steering-delay integration, plane-wave fallback,
and Bartlett beamforming. Its stored feature is the actual Green estimate.
Regenerate older RBD data that saved the FFT return value under that name.

| Setting | Choices and effect |
|---|---|
| `rbd.beam_selection` | `best`: strongest beam; `multipath`: phase-rotated, beam-power-weighted accepted components |
| `rbd.frequency_estimation` | `selected`: selected bins; `full`: all one-sided FFT bins |

Standalone `full` generation can leave other routes on selected frequencies.
The remote comparison workflow therefore requires shared modes `["full"]`
when RBD uses `full`, so all routes align.

RBD divides each spectrum vector by its norm plus float64 machine epsilon.
Norms well above epsilon approach one. With an expected generation
configuration, the validator applies these checks:

| Beam mode | Green-feature norm requirement |
|---|---|
| `best` | Columns above `1e-8` must be within `2e-6` of unit norm; DC is exempt in full mode |
| `multipath` | Columns must not exceed `1+2e-6` |

These checks distinguish Green features from raw FFT amplitudes; they do not
certify the physical estimator.

RBD diagnostics are stored under `/rbd`: theta grid/best/selected angles,
accepted angle count/powers, full beam powers and spectrum scale. Unused arrival
slots are NaN; model input `/X` remains finite. Shapes and peak-selection
behavior are defined in [RBD multipath](../methods/rbd-multipath-detection.md).

## Windows, Frequencies, and Splits

### Window Geometry

Dataset configuration controls `segment_duration_s`, `segment_step_s`,
`segment_start_s`, `segment_end_s`, `num_snapshots_per_segment` and
`snapshot_overlap_count`. ELM/SCM combine Ns consecutive base snapshots and
advance by `(Ns-overlap)` base windows. Their labels/times describe the combined
window. RBD operates on base windows.

Remote comparison suites require Ns=1, no overlap, and disjoint base windows
so all routes evaluate identical rows.

### Frequency Selection

Frequency modes are arrays drawn from the choices below. `full` must be used alone.

| Mode | Default selection |
|---|---|
| `full` | All one-sided FFT bins |
| `mel` | 64 bins |
| `deep` | `[49,64,79,94,112,130,148,166,201,235,283,338,388]` Hz targets |
| `shallow` | `[109,127,145,163,198,232,280,335,385]` Hz targets |
| `adapt` | 16 bins fitted on valid training windows only |

Targets map to the nearest one-sided FFT bins; duplicates are removed and
the axis is sorted. Adapt freezes one dataset-level axis for held-out testing.
Standalone generation allows overrides in `dataset.frequency_selection_config`.

### Train and Test Splits

| Strategy | Rule |
|---|---|
| `full_interleaved` | Use the complete configured candidate interval, with deterministic interleaving by `train_test_ratio`, default 4:1 |
| `range_overlap` | Keep the shared range interval of candidate windows before/after closest approach; `train_side` selects the training side |

`range_overlap` locates the minimum valid window-center range, excludes that
window and `floor(gap_s/window_step_s)` neighboring windows on each side,
and takes up to `half_duration_s` of candidate windows per side. Each side is
clipped independently by the recording boundaries; invalid/nonfinite windows
are excluded independently. For candidate-side ranges `[Lmin,Lmax]` and
`[Rmin,Rmax]`, the shared interval is
`[max(Lmin,Rmin), min(Lmax,Rmax)]`. Only samples whose center-range labels fall
inside that inclusive interval enter the split. An empty side, an interval of
zero/negative width, or a side with no sampled labels inside the intersection
is rejected.

Side counts may differ. This is interval overlap, without one-to-one distance
matching or equal label distributions. `train_test_ratio` is unused;
`train_side=before` trains before closest approach and tests after it, while
`after` reverses the sides. Optional `test_step_s` subsamples the selected test
side chronologically after range filtering. Metadata records
`range_overlap_min_km`, `range_overlap_max_km`, candidate/selected side counts,
and the selected time bounds. The earlier Python names `periodic` and
`Range_nearby` are rejected; static MATLAB references retain their original names.

Dataset names encode the route, estimator, and window/frequency settings.
`manual_dataset_variant_tag` adds a safe variant suffix before the configuration
fingerprint.

For model operations, see [training and prediction](../workflows/training-and-prediction.md).
