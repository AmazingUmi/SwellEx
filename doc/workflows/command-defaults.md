# Workflow Configuration Defaults

Tables below describe the complete
[base TOML preset](../../scripts/base_pack/localization_networks/workflow/haiqin_full_interleaved.toml),
validated by [workflow/config.py](../../scripts/base_pack/localization_networks/workflow/config.py).
They are workflow settings, not direct Python CLI defaults. Joint presets
use the same numerical defaults and require an additional `[fusion]` table.
`MoE_learn` uses a separate short `demo` preset and staged-training schema;
its [pack guide](../packs/moe-learn.md) defines those defaults.

Copy the selected pack's preset, edit existing fields, and pass `--config PATH`.
Unknown or missing tables/fields fail. Epochs and seed are configured here; the
effective configuration is saved in `request.json`.
See [shell options and examples](remote-validation.md#experiment-parameters).

## Execution

| Parameter | Default | Meaning / constraint |
|---|---|---|
| `schema_version` | `1` | Required configuration version |
| `remote.host` | `haiqin` | SSH alias or user@host |
| `validation.device` | `cuda` | Requested device; unavailable CUDA fails |
| `validation.seed` | `2026` | Integer `0..4294967295` |
| `validation.threads` | `4` | Positive CPU thread count |
| `validation.quick_timeout_s` | `900` | Timeout per check or aggregate-render command, seconds |
| `validation.generation_timeout_s` | `1800` | Generation timeout, seconds |
| `validation.training_timeout_s` | `7200` | Timeout per train/build/predict command, seconds |

Remote path parameters are listed once in the
[environment guide](remote-environment.md#configured-paths-and-runtime).
Timeouts stop the command's process group.

## Dataset

| Parameter in `[dataset]` | Default | Meaning / constraint |
|---|---|---|
| `event_name` | `S5` | Workflow supports S5 |
| `fs` | `1500` | Sample rate in Hz |
| `segment_start_s` | `0` | First candidate window start, seconds |
| `segment_end_s` | `4499` | Last candidate window **start**, inclusive; not the recording end |
| `segment_duration_s` | `1.0` | Base window duration, seconds |
| `segment_step_s` | `1.0` | Window step; at least the duration for remote comparisons |
| `num_snapshots_per_segment` | `1` | Remote comparisons require Ns=1 |
| `snapshot_overlap_count` | `0` | Remote comparisons require no overlap |
| `frequency_selection_modes` | `["deep", "shallow"]` | Unique modes; `full` must be used alone |
| `split_strategy` | `full_interleaved` | `full_interleaved` or `range_overlap` |
| `train_test_ratio` | `[4, 1]` | Positive train/test counts per cycle; unused for range overlap |
| `train_side` | `before` | Range overlap only; `before` or `after` closest approach |
| `half_duration_s` | `800` | Range overlap only; candidate duration per side |
| `gap_s` | `15` | Range overlap only; excluded neighborhood around closest approach |
| `test_step_s` | `1` | Range overlap test subsampling step, seconds |
| `manual_dataset_variant_tag` | `""` | Optional letters/digits/underscore/hyphen suffix |

Full S5 with these settings yields 4500 windows: 3600 train and 900 test.
The neural 25% internal validation split leaves 2700 optimizer-training rows.
All three routes are prepared even for one selected model.
Window geometry, frequency modes, and split rules are defined in the
[HDF5 guide](../datasets/hdf5-datasets-and-splits.md).

| Parameter in `[rbd]` | Default | Meaning / constraint |
|---|---|---|
| `beam_selection` | `best` | Strongest beam; `multipath` aggregates accepted peaks |
| `frequency_estimation` | `selected` | `selected` or `full`; full requires shared modes `["full"]` |
| `use_plane_wave` | `false` | Enable plane-wave steering instead of sound-speed integration |
| `normalize_spectrum` | `false` | Apply spectrum scaling before beam decomposition |

The workflow uses the generator's default multipath thresholds. Custom
`multipath_options` and `frequency_selection_config` belong to standalone
**generation JSON**; they are not accepted workflow TOML fields.

## Neural Training and Prediction

| Parameter in `[training]` | Default | Meaning / constraint |
|---|---|---|
| `epochs` | `100` | Positive epoch count per neural model |
| `batch_size` | `64` | Positive train and prediction batch size |
| `base_channels` | `16` | CNN base width; ResNet uses its own channel layout |
| `loss_space` | `km` | `km` or `normalized` SmoothL1 loss |
| `huber_beta` | `0.5` | Positive transition value in the chosen loss space |
| `val_fraction` | `0.25` | Internal validation fraction when no validation files exist; strictly between 0 and 1 |

Workflow runs start fresh and predict with their own best checkpoint. They
inherit learning rate `3e-4`, weight decay `1e-4`, and dropout `0.15` from the
Python CLI, and explicitly use zero DataLoader workers. These are not workflow
TOML fields. Direct Python instead defaults to train/predict batches 16/32
and normalized loss; see [manual parameters](training-and-prediction.md#train-options).

## GRNN

| Parameter in `[grnn]` | Default | Meaning / constraint |
|---|---|---|
| `cv_folds` | `5` | Integer at least 2; training-reference CV |
| `sigma_candidates` | `0.01..0.10` by `0.01`, then `0.20..2.00` by `0.10` | 29 positive candidates; minimize mean fold MAPE |
| `batch_size` | `1024` | Positive query batch size for build/predict |

Each selection builds its own reference. Feature-coordinate standardization
and per-sample channel normalization are off. GRNN has no epochs or optimizer.
Direct Python defaults to fixed sigma without CV and batch 128; see
[manual build parameters](standalone-grnn.md#build-options).

## Joint Fusion

| Parameter | Default | Meaning / constraint |
|---|---|---|
| `[fusion] mode` | `block_rms` | Joint only; `block_rms` or `raw_concat`; shell `--fusion` takes precedence |

Fitted scales, fold isolation, and artifact identity are specified in the
[joint feature contract](../packs/joint-feature-pack.md#representation-and-fitted-scales).

## A5000 Sizing

Batch 64 and four CPU threads are the maintained base settings, originally
chosen for haiqin's RTX A5000. They are not a measured memory guarantee for
joint inputs or competing GPU workloads. Inspect the current run's environment
and memory use before increasing them; this configuration does not establish
an optimal batch size for accuracy.
