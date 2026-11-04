# HDF5 Datasets and Split Strategies

## Generation

Each pack's pure-CPU generator is `scripts/data_processing/<pack>/generate.py --config PATH` (`generation_config.py` validates strict JSON fields; `generation_core.py` handles windows/splits/HDF5; `feature_math.py` computes features; `validate.py` verifies numerical/HDF5 invariants, backed by shared helpers in `scripts/data_processing/common/`). `scripts/data_processing` never imports PyTorch, matplotlib, `network_training`, or `remote_workflow`. Raw inputs are VLA channel MAT files, range/position text tracks, and CTD profiles (`Origindata/environments/ctds/i9605.prn`). Static MATLAB references in `scripts/data_processing/matlab_reference/` are never executed.

- **Mean centering**: `MoE_learn` subtracts each element's mean **within each window** (`moe-learn-python-features-v3-window-centered`); the four earlier packs subtract each element's mean across the entire loaded recording prior to windowing. Adaptive frequency selection (`adapt`) always fits on valid training windows only.
- **Outputs**: Written atomically via temporary files under `outputs/Datasets/<name>_<12hex_fingerprint>/` (refusing existing directories unless a new `manual_dataset_variant_tag` is set), alongside JSON provenance and raw-input SHA256 hashes.

## Stored Contracts

`/X` is `float32` in logical axis order (with `axis_order` / `storage_axis_order` attributes; loaders also handle reversed MATLAB axis order when complete metadata is present). Vector fields have shape `[N, 1]`:

```text
/y_range_km, /valid_sample
/time/{window_start_s, window_center_s, window_stop_s, sample_start_idx, sample_stop_idx}
/frequency/freq_hz                    strictly increasing, finite >= 0 (length F)
/array/{element_idx, depth_m}         1..E, strictly increasing depths >= 0
/pair/{numerator_element_idx, denominator_element_idx}   1-based column-major (ELM/SCM)
/range/{range_time_s, range_km_raw}
/split/source_segment_idx             1-based window identity
```

Loaders drop invalid/nonfinite labels, reject nonfinite `/X`, and compare `input_signature` (frequencies, element/depth axes, and pair sequences) across train/val/test files and checkpoints/references.

## Feature Routes

| Route | Logical `/X` | Network input | Definition |
|---|---|---|---|
| `RBD` | `[S, E, F, 2]` | `[batch, 2, E, F]` | Bartlett Green estimate (`best` strongest beam or `multipath` weighted peaks; unit-norm spectrum columns within `2e-6`, diagnostics in `/rbd`; see [RBD multipath](rbd-multipath-detection.md)) |
| `ELM` | `[S, P, F, 2]` | `[batch, 2, P, F]` | Strict-upper (`i<j`, $P=E(E-1)/2$) frequency ratios: $\sum_s X_{i,s}\overline{X_{j,s}} / (\sum_s |X_{j,s}|^2 + 10^{-6}\text{median\_power} + \varepsilon)$ |
| `SCM` | `[S, P, F, 2]` | `[batch, 2, P, F]` | Upper-triangle + diagonal (`i<=j`, $P=E(E+1)/2$) averaged unit-snapshot outer products (norm floor $10^{-12}$; see [GRNN note](../network_training/grnn.md)) |

Channel order on the complex axis is `(real, imag)`; ELM/SCM pairs are stored 1-based in column-major order. Concatenated multi-route inputs (`joint_feature_pack`, `MoE_data_scale`, `MoE_learn`) follow [joint feature concatenation](joint-features.md).

## Windows, Frequencies, and Splits

- **Windows**: `segment_duration_s` and `segment_step_s` must align to integer samples within inclusive start bounds `[segment_start_s, segment_end_s]`. ELM/SCM combine `num_snapshots_per_segment` (`Ns`) snapshots advancing by `Ns - snapshot_overlap_count`; remote cross-method comparisons and joint packs require `Ns=1`, `overlap=0`, `step >= duration`, and shared frequency bins (`rbd.frequency_estimation="full"` requires `frequency_selection_modes=["full"]`).
- **Frequency modes**: Unique entries from `full` (used alone), `mel` (64 bins), `deep` (`[49,64,79,94,112,130,148,166,201,235,283,338,388]` Hz), `shallow` (`[109,127,145,163,198,232,280,335,385]` Hz), and `adapt` (16 bins fitted on training windows), snapped to sorted unique one-sided FFT bins.

### Train and Test Splits

| Strategy | Rule |
|---|---|
| `full_interleaved` | Deterministic cycle over candidate windows using `train_test_ratio` (default `[4, 1]`) |
| `range_overlap` | Exclude closest-approach window ± `floor(gap_s / step_s)` windows, take up to `half_duration_s` per side, keep windows in shared range `[max(Lmin, Rmin), min(Lmax, Rmax)]`, assign `train_side` (`before` or `after`), and optionally subsample test chronologically by `test_step_s` |

`range_overlap` guarantees range-interval intersection (not equal counts or identical label distributions) and rejects empty or non-overlapping sides.

### Continual Stage and Distance-Holdout Splits

`MoE_learn` partitions aligned `RBD_SCM` windows into staged streams (`scripts/network_training/MoE_learn/data.py`; training logic in [continual MoE](../network_training/moe-learn.md)):

- **`distance_holdout`** (`distance_edges_km = [0.90, 1.32, 1.74, 2.16]`, `heldout_group = 1|2|3`): Reserves one complete after-side distance interval (`near` `[0.90, 1.32)`, `middle` `[1.32, 1.74)`, or `far` `[1.74, 2.16] km`) as fixed test data and **excludes that interval from all before/after training, validation, and replay**. Excluded before-side window identities are persisted, and preprocessing fits strictly after those exclusions. Remaining before rows split 80:20 for stage 1; remaining after rows split 80:20 and arrive ordered near-to-far across stages 2 and 3.
- **`before_then_after`** (`split_strategy="range_overlap"`, `train_side="before"`, `after_test_fraction=0.2`): Stage 1 uses the 800 before windows (`640` train / `160` val). From 700 after windows, `140` form a fixed test set covering all distances, and the remaining `560` (`448` train / `112` val) arrive in stages 2–4 either `time_interleaved` or grouped by `range_bins` (unlike `distance_holdout`, test distances are already present in stage 1).
- **`interleaved` / `chronological`**: Deterministically partition `full_interleaved` training-source windows by stride or contiguous time blocks.
