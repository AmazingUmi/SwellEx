# Workflow Configuration Defaults

Tables below document [`presets/full.toml`](../../scripts/remote_workflow/presets/full.toml), validated by [`remote_workflow/config.py`](../../scripts/remote_workflow/config.py). Copy a bundled preset from `scripts/remote_workflow/presets/` (`full`, `train_before`, `train_after`), edit fields, and pass `--config PATH` (unknown or missing keys fail).

## Execution

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `schema_version` | `1` | Required schema version |
| `remote.host` | `haiqin` | SSH host; path fields are in [remote environment](remote-environment.md#configured-paths-and-runtime) |
| `validation.device` | `cuda` | Required device; missing CUDA fails without CPU fallback |
| `validation.seed` / `threads` | `2026` / `4` | Seed in `0..2**32-1`; positive CPU thread count |
| `validation.{quick,generation,training}_timeout_s` | `900` / `1800` / `7200` | Per-command process-group timeouts (seconds) |

## Dataset and RBD

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `dataset.event_name` / `fs` | `S5` / `1500` | Workflow supports event `S5` at 1500 Hz |
| `dataset.segment_start_s` / `segment_end_s` | `0` / `4499` | Inclusive first and last window **start** times (s); yields 4500 windows |
| `dataset.segment_duration_s` / `segment_step_s` | `1.0` / `1.0` | Disjoint 1 s windows (`step >= duration` for remote comparisons) |
| `dataset.num_snapshots_per_segment` / `snapshot_overlap_count` | `1` / `0` | Remote comparisons require `Ns=1` and zero overlap |
| `dataset.frequency_selection_modes` | `["deep", "shallow"]` | Unique modes (`full` must be used alone); see [HDF5 guide](../data_processing/hdf5-datasets-and-splits.md) |
| `dataset.split_strategy` / `train_test_ratio` | `full_interleaved` / `[4, 1]` | `full_interleaved` (3600 train / 900 test) or `range_overlap` |
| `dataset.{train_side,half_duration_s,gap_s,test_step_s}` | `before` / `800` / `15` / `1` | `range_overlap` parameters |
| `dataset.manual_dataset_variant_tag` | `""` | Safe alphanumeric/`_`/`-` cache-variant tag |
| `rbd.{beam_selection,frequency_estimation}` | `best` / `selected` | `best` or `multipath`; `selected` or `full` (`full` requires `["full"]` modes) |
| `rbd.{use_plane_wave,normalize_spectrum}` | `false` / `false` | Plane-wave steering fallback; pre-beamforming spectrum scaling |

Custom `rbd.multipath_options` and `frequency_selection_config` belong only to standalone generation JSON, not workflow TOML.

## Neural Training and GRNN

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `training.{epochs,batch_size,base_channels}` | `100` / `64` / `16` | Positive integers (`ResNet` uses fixed torchvision widths; CLI batch defaults to `16`/`32`) |
| `training.{loss_space,huber_beta,val_fraction}` | `km` / `0.5` / `0.25` | `km` or `normalized` SmoothL1 (CLI defaults to `normalized`); holdout in `(0, 1)` |
| `grnn.{cv_folds,batch_size,sigma_candidates}` | `5` / `1024` / 29 grid values | `0.01..0.10` (`0.01`) and `0.20..2.00` (`0.10`) |

Workflow neural runs inherit `lr=3e-4`, `weight_decay=1e-4`, `dropout=0.15`, and `num_workers=0` from the Python CLI.
