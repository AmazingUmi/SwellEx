# Shell Command Defaults

These values apply to `./scripts/run.sh METHOD ARCH` with the bundled
[haiqin.toml](../../scripts/localization_networks/workflow/haiqin.toml).
`METHOD` and `ARCH` are required and have no defaults.

Use `--epochs` or `--seed` to override those two values. For other TOML
settings, copy the complete file and pass `--config PATH`. The merged settings
are saved in `request.json`; see the [command guide](remote-validation.md).

## Execution

| Setting | Default |
|---|---|
| `remote.host` | `haiqin` |
| `validation.device` | `cuda`; the requested device must be available |
| `validation.seed` | `2026` |
| `validation.threads` | `4` |
| `validation.quick_timeout_s` | `600` s |
| `validation.generation_timeout_s` | `900` s |
| `validation.training_timeout_s` | `900` s for each train/build/predict command |
| Foreground waiting and automatic fetch | On |
| Dataset cache reuse | On, with hashes and validation checked |
| Dry run | Off |
| `validate` suite | `smoke` |

The TOML `schema_version` is `1`. Remote paths and the interpreter are listed in
the [frozen environment guide](remote-environment.md#configured-paths-and-runtime).

## Dataset

The bundled interval is a small pipeline check. The end value is the last
window's **start time**, not its end time.

| Field in `[dataset]` | Default |
|---|---|
| `event_name` | `S5` |
| `fs` | `1500` Hz |
| `segment_start_s` | `0` s |
| `segment_end_s` | `59` s; window starts include both endpoints |
| `segment_duration_s` | `1.0` s |
| `segment_step_s` | `1.0` s |
| `num_snapshots_per_segment` | `1` |
| `snapshot_overlap_count` | `0` |
| `frequency_selection_modes` | `["deep", "shallow"]` |
| `split_strategy` | `periodic` |
| `train_test_ratio` | `[4, 1]`, train:test |
| `train_side` | `before`; used only by `Range_nearby` |
| `half_duration_s` | `800` s; used only by `Range_nearby` |
| `gap_s` | `15` s; used only by `Range_nearby` |
| `test_step_s` | `1` s; used only by `Range_nearby` |
| `manual_dataset_variant_tag` | Empty string |

All three feature routes are generated or validated even when one model is
selected. For field semantics, see
[HDF5 datasets and splits](../datasets/hdf5-datasets-and-splits.md).

| Field in `[rbd]` | Default |
|---|---|
| `beam_selection` | `best` |
| `frequency_estimation` | `selected` |
| `use_plane_wave` | `false` |
| `normalize_spectrum` | `false` |

## Neural Training and Prediction

Each selected run starts fresh and predicts with its own best checkpoint.

| Field in `[training]` | Default |
|---|---|
| `epochs` | `2` |
| `batch_size` | `8`, for both train and predict |
| `base_channels` | `8` for CNNs; ResNet uses its architecture's channel layout |
| `loss_space` | `km` |
| `huber_beta` | `0.5` km |
| `val_fraction` | `0.25`, used when no validation files exist |

The worker also inherits these direct network CLI defaults:

| Network setting | Effective default |
|---|---|
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Dropout | `0.15` |
| DataLoader workers | `0`, explicitly supplied by the worker |
| Per-sample input channel normalization | On for complex CNN/ResNet; off for real CNN |

Learning rate, weight decay, dropout, workers, and input normalization are not
workflow TOML fields or `run.sh` options. Manual Python commands have different
defaults for epochs, batch size, channels, and loss space; see
[training and prediction](training-and-prediction.md#train-options).

## SCM-GRNN

GRNN builds a fresh reference and uses it explicitly for prediction. It has
no epochs or optimizer and uses `validation.seed` for cross-validation.

| Field in `[grnn]` | Default |
|---|---|
| `cv_folds` | `3` |
| `sigma_candidates` | `[0.01, 0.1, 1.0]` |
| `batch_size` | `64`, for both build and predict |

Feature-coordinate standardization and per-sample channel normalization are
both off. The direct GRNN CLI uses different defaults; see
[manual build options](standalone-scm-grnn.md#build-options).

## Result Delivery

| Output | Default |
|---|---|
| Logs, tables, and reports | Local `output/data/<run_id>/` |
| Figures | Local `output/figure/<run_id>/` |
| Aggregate figure formats | PNG at 300 dpi and PDF |
| Original per-model plots | PNG at 200 dpi |
| HDF5 datasets and `.pt` artifacts | Kept on haiqin |

See [result visualization](result-visualization.md) for the directory layout
and figure definitions.
