# Workflow Configuration Defaults

Tables below document [`haiqin_full_interleaved.toml`](../../scripts/remote_workflow/base_pack/haiqin_full_interleaved.toml), validated by [`remote_workflow/base_pack/config.py`](../../scripts/remote_workflow/base_pack/config.py) (and per-pack `config.py`). Copy a bundled preset from `scripts/remote_workflow/<pack>/`, edit fields, and pass `--config PATH` (unknown or missing keys fail).

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

## Neural Training, GRNN, and Pack Extensions

| Parameter | Default | Constraint / meaning |
|---|---|---|
| `training.{epochs,batch_size,base_channels}` | `100` / `64` / `16` | Positive integers (`ResNet` uses fixed torchvision widths; CLI batch defaults to `16`/`32`) |
| `training.{loss_space,huber_beta,val_fraction}` | `km` / `0.5` / `0.25` | `km` or `normalized` SmoothL1 (CLI defaults to `normalized`); holdout in `(0, 1)` |
| `training.train_fraction` | `1.0` | `data_scale_pack` / `MoE_data_scale`: seeded 32-row block subsampling in `(0, 1]` |
| `scaling.*` / `moe.balance_loss_weight` | Pack defaults | Capacity/expert rules in [data scale](../network_training/data-scale-pack.md) and [MoE data scale](../network_training/moe-data-scale.md) |
| `grnn.{cv_folds,batch_size,sigma_candidates}` | `5` / `1024` / 29 grid values | `0.01..0.10` (`0.01`) and `0.20..2.00` (`0.10`); `base_pack` and `joint_feature_pack` only |
| `fusion.mode` | `block_rms` | `joint_feature_pack` only: `block_rms` or `raw_concat` (`--fusion` overrides) |

Workflow neural runs inherit `lr=3e-4`, `weight_decay=1e-4`, `dropout=0.15`, and `num_workers=0` from the Python CLI.

## Continual MoE Staged Defaults

`scripts/remote_workflow/MoE_learn/` bundles `--preset demo` ([`haiqin_demo.toml`](../../scripts/remote_workflow/MoE_learn/haiqin_demo.toml)) plus full `--config` protocols (`haiqin_holdout_{near,middle,far}.toml`, `haiqin_before_then_after.toml`, `haiqin_before_then_after_range.toml`; training rules in [continual MoE](../network_training/moe-learn.md)):

| Demo parameter | `haiqin_demo.toml` | Full TOMLs / role |
|---|---|---|
| Interval / split / stages | S5 `0–119 s` / `full_interleaved` 4:1 / 2 `interleaved` stages | S5 `0–4499 s`; `distance_holdout` (3 stages) or `before_then_after` (4 stages, `val_fraction=0.2`) |
| Epochs (`init` / `adapt` / `warmup`) | `3 / 2 / 1` | `30 / 30 / 30`; batch `8`, base channels `16` |
| Initial / max experts | `2 / 4` | Adds at most 1 expert per accepted stage expansion |
| Replay / history guard | `buffer` (`16`) / `true` | Full configs use `replay_mode="full"` and `history_validation=true` |
| Acceptance / guard / novelty | `0.01` / `0.05` / `3.0` | `min_relative_gain`, `max_old_relative_regression`, `novelty_threshold` |
| Learning rates (`init` / `old` / `new` / `router`) | `1e-3` / `1e-4` / `1e-3` / `1e-3` | Positive finite rates requiring `old_expert_learning_rate < new_expert_learning_rate` |
