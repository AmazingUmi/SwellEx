# HDF5 Datasets and Split Strategies

## Generation

The pure-CPU generator is `scripts/data_processing/generate.py --config PATH [--method {RBD,SCM} ...]` (`generation_config.py` is the canonical stdlib schema validator for `[dataset]`, `[rbd]`, and `[scm]`; `generation_core.py` handles windows/splits/HDF5; `feature_math.py` computes features; `validate.py [datasets --methods ...]` verifies numerical/HDF5 invariants; all modules sit directly in `scripts/data_processing/`). `scripts/data_processing` never imports PyTorch, matplotlib, `network_training`, or `remote_workflow`. Raw inputs are VLA channel MAT files (element count read dynamically from `positions_vla.txt`), range/position text tracks, and CTD profiles (`Origindata/environments/ctds/<ctd_file>`, loaded only when `RBD` is requested). Static MATLAB references in `scripts/data_processing/matlab_reference/` are never executed.

- **Centering and FFT windowing**: `dataset.signal_centering` supports `"full_record"` (preset default, per-element mean subtracted across the loaded recording), `"per_window"` (subtracted per snapshot/window before tapering), and `"none"`. `dataset.fft_window` supports `"boxcar"` (preset default) and `"hann"` (`np.hanning`). Adaptive frequency selection (`adapt`) always fits on valid training windows only (`adapt_max_num_segments = 0` uses all valid training windows).
- **Outputs**: Written atomically via temporary files under `outputs/Datasets/<name>_<12hex_fingerprint>/` (with per-method fingerprints so `[rbd]`/`ctd_file` never invalidate `SCM` and `[scm]` never invalidates `RBD`), alongside JSON provenance and raw-input SHA256 hashes.

## Stored Contracts

`/X` is `float32` in logical axis order (with `axis_order` / `storage_axis_order` attributes; loaders also handle reversed MATLAB axis order when complete metadata is present). Vector fields have shape `[N, 1]`:

```text
/y_range_km, /valid_sample
/time/{window_start_s, window_center_s, window_stop_s, sample_start_idx, sample_stop_idx}
/frequency/freq_hz                    strictly increasing, finite >= 0 (length F)
/array/{element_idx, depth_m}         1..E, strictly increasing depths >= 0
/pair/{numerator_element_idx, denominator_element_idx}   1-based column-major (SCM)
/range/{range_time_s, range_km_raw}
/split/source_segment_idx             1-based window identity
```

Loaders drop invalid/nonfinite labels, reject nonfinite `/X`, and compare `input_signature` (frequencies, element/depth axes, and pair sequences) across train/val/test files and checkpoints/references.

## Feature Routes

| Route | Logical `/X` | Network input | Definition |
|---|---|---|---|
| `RBD` | `[S, E, F, 2]` | `[batch, 2, E, F]` | Bartlett Green estimate over `rbd.{angle_min_deg,angle_max_deg,num_angles}` (`best` strongest beam or `multipath` weighted peaks; unit-norm spectrum columns within `2e-6`, diagnostics in `/rbd`; see [RBD multipath](rbd-multipath-detection.md)) |
| `SCM` | `[S, P, F, 2]` | `[batch, 2, P, F]` | Upper-triangle + diagonal (`i<=j`, $P=E(E+1)/2$) averaged unit-snapshot outer products over `[scm]` (`num_snapshots_per_segment`, `snapshot_overlap_count`, `norm_floor`; see [GRNN note](../network_training/grnn-method.md)) |

Channel order on the complex axis is `(real, imag)`; SCM pairs are stored 1-based in column-major order. ELM and concatenated multi-route inputs are archived (see [archive](../../archive/README.md)).

## Windows, Frequencies, and Splits

- **Windows**: `segment_duration_s` and `segment_step_s` must align to integer samples within inclusive start bounds `[segment_start_s, segment_end_s]` (`-1.0` / `null` selects automatic recording bounds). Under `[scm]`, SCM combines `num_snapshots_per_segment` (`Ns >= 1`) base snapshots advancing by `Ns - snapshot_overlap_count` (`0 <= overlap < Ns`). `full_interleaved` requires non-overlapping windows to prevent train/test raw-sample leakage; `range_overlap` supports overlapping base segments and overlapping multi-snapshot SCM windows (`gap_s` must keep train/test samples disjoint across closest approach). Cross-route equivalence is enforced when both routes share identical window grids (`Ns=1`, `overlap=0`, matching frequency mode), while multi-snapshot or single-route runs validate each route independently.
- **Frequency modes**: Unique entries from `full` (used alone), `mel` (64 bins), `deep` (`[49,64,79,94,112,130,148,166,201,235,283,338,388]` Hz), `shallow` (`[109,127,145,163,198,232,280,335,385]` Hz), and `adapt` (16 bins fitted on training windows), snapped to sorted unique one-sided FFT bins.

### Train and Test Splits

| Strategy | Rule |
|---|---|
| `full_interleaved` | Deterministic cycle over candidate windows using `train_test_ratio` (default `[4, 1]`) |
| `range_overlap` | Exclude closest-approach window ± `floor(gap_s / step_s)` windows, take up to `half_duration_s` per side, keep windows in shared range `[max(Lmin, Rmin), min(Lmax, Rmax)]`, assign `train_side` (`before` or `after`), and optionally subsample test chronologically by `test_step_s` |

`range_overlap` guarantees range-interval intersection (not equal counts or identical label distributions) and rejects empty or non-overlapping sides.
