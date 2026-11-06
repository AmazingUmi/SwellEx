# Workflow Configuration Defaults

Tables below document [`presets/full.toml`](../../scripts/remote_workflow/presets/full.toml), validated by [`remote_workflow/config.py`](../../scripts/remote_workflow/config.py) and [`data_processing/generation_config.py`](../../scripts/data_processing/generation_config.py). Copy a bundled preset from `scripts/remote_workflow/presets/` (`full`, `train_before`, `train_after`), edit fields, and pass `--config PATH` (all 9 top-level tables and their documented subtables are required; unknown or missing keys fail). Every distinct configuration produces isolated dataset cache keys and run outputs.

## Experiment and Execution

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `schema_version` | `1` | Required schema version |
| `experiment.methods` | `["RBD", "SCM"]` | Non-empty unique list (`len >= 1`) from `RBD`, `SCM`; overridden by CLI `--method` |
| `experiment.networks` | `["complex_CNN", "ResNet18", "ResNet50", "GRNN"]` | Non-empty unique list (`len >= 1`); overridden by CLI `--network` |
| `remote.host` | `haiqin` | SSH host; path fields (`repository`, `workspace`, `base_python`, `uv`) are in [remote environment](remote-environment.md#configured-paths-and-runtime) |
| `validation.device` | `cuda` | `cpu`, `cuda`, or `cuda:N`; missing CUDA fails without CPU fallback |
| `validation.seed` / `threads` | `2026` / `4` | Seed in `0..2**32-1`; positive integer CPU thread count |
| `validation.{quick,generation,training}_timeout_s` | `900` / `1800` / `7200` | Per-command process-group timeouts (seconds) |

## Dataset, RBD, and SCM

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `dataset.event_name` / `fs` | `"S5"` / `1500` | Event matching `S\d+`; positive integer sample rate (Hz) |
| `dataset.ctd_file` | `"i9605.prn"` | Safe CTD filename under `Origindata/environments/ctds/` (loaded and hashed only when `RBD` is active) |
| `dataset.segment_start_s` / `segment_end_s` | `0.0` / `4499.0` | Inclusive first/last base-window **start** times (s); use `-1.0` for automatic recording bounds |
| `dataset.segment_duration_s` / `segment_step_s` | `1.0` / `1.0` | Base snapshot length and step (s), integer sample counts (`full_interleaved` requires non-overlapping windows; `range_overlap` supports `step < duration`) |
| `dataset.signal_centering` / `fft_window` | `"full_record"` / `"boxcar"` | `"full_record"`, `"per_window"`, or `"none"`; `"boxcar"` or `"hann"` |
| `dataset.frequency_selection_modes` | `["deep", "shallow"]` | Non-empty unique list from `full` (used alone), `mel`, `deep`, `shallow`, `adapt`; see [HDF5 guide](../data_processing/hdf5-datasets-and-splits.md) |
| `dataset.frequency_selection_config.*` | Preset defaults | `mel_{num_bins,min_freq_hz,max_freq_hz}`, `{deep,shallow}_target_freq_hz`, `adapt_{num_bins,min_freq_hz,max_freq_hz,max_num_segments}` (`0` uses all valid training windows) |
| `dataset.split_strategy` / `train_test_ratio` | `"full_interleaved"` / `[4, 1]` | `"full_interleaved"` or `"range_overlap"` |
| `dataset.{train_side,half_duration_s,gap_s,test_step_s}` | `"before"` / `800.0` / `15.0` / `1.0` | `range_overlap` parameters; set `test_step_s = 0.0` to inherit each method's effective window step |
| `dataset.manual_dataset_variant_tag` | `""` | Safe alphanumeric/`_`/`-` variant tag |
| `rbd.{beam_selection,frequency_estimation}` | `"best"` / `"selected"` | `"best"` or `"multipath"`; `"selected"` or `"full"` (checked when `RBD` is active) |
| `rbd.{use_plane_wave,normalize_spectrum}` | `false` / `false` | Plane-wave steering delays; pre-beamforming spectrum normalization |
| `rbd.{angle_min_deg,angle_max_deg,num_angles}` | `-90.0` / `90.0` / `181` | Grazing-angle grid in `[-90, 90]` with `angle_min_deg < angle_max_deg` and `num_angles >= 2` |
| `rbd.multipath_options.*` | `-6.0` / `2.0` / `181` / `3.0` | `peak_threshold_db`, `min_separation_deg`, `max_num_peaks`, `sidelobe_reject_db` |
| `scm.num_snapshots_per_segment` / `snapshot_overlap_count` | `1` / `0` | Multi-snapshot SCM (`Ns >= 1`, `0 <= overlap < Ns`; overlapping windows require `range_overlap`) |
| `scm.norm_floor` | `1e-12` | Positive snapshot spectral norm floor (`<= 1.0`) |

## Neural Training and GRNN

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `training.{epochs,batch_size,predict_batch_size}` | `100` / `64` / `64` | Positive integers |
| `training.{lr,weight_decay}` | `3e-4` / `1e-4` | Positive learning rate; nonnegative AdamW weight decay |
| `training.{base_channels,dropout}` | `16` / `0.15` | Positive integer CNN width (`ResNet` uses fixed torchvision widths); dropout in `[0, 1)` |
| `training.{loss_space,huber_beta,normalize_input}` | `"km"` / `0.5` / `true` | `"km"` or `"normalized"` SmoothL1; nonnegative beta; per-sample real/imag z-score |
| `training.{val_split_mode,val_fraction,val_gap_s}` | `"random"` / `0.25` / `0.0` | `"random"` seeded holdout or `"blocked"` chronological trailing holdout with temporal buffer `val_gap_s >= 0` |
| `training.num_workers` | `0` | Nonnegative DataLoader worker count |
| `grnn.{cv_folds,spread,sigma_candidates}` | `5` / `0.01` / 29 grid values | `cv_folds = 0` uses fixed `spread` directly; `cv_folds >= 2` requires non-empty `sigma_candidates` |
| `grnn.{standardize_input,per_sample_channel_norm}` | `false` / `false` | Per-coordinate reference z-score; per-sample real/imag normalization |
| `grnn.{batch_size,predict_batch_size}` | `1024` / `1024` | Positive integer evaluation block sizes |
| `grnn.max_{reference,validation,predict}_samples` | `0` / `0` / `0` | Nonnegative sample caps (`0` keeps all valid windows) |
