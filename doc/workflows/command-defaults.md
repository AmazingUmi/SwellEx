# Shell Command Defaults

These values apply to `./scripts/run.sh METHOD ARCH` with the bundled
[haiqin_full_interleaved.toml](../../scripts/localization_networks/workflow/haiqin_full_interleaved.toml).
`METHOD` and `ARCH` are required and have no defaults.

Use `--epochs` or `--seed` to override those two values. For other TOML
settings, copy the complete file and pass `--config PATH`. The merged settings
are saved in `request.json`; see the [command guide](remote-validation.md).
The two independent `range_overlap` configurations are listed under
[split configurations](remote-validation.md#split-configurations).

## Execution

| Setting | Default |
|---|---|
| `remote.host` | `haiqin` |
| `validation.device` | `cuda`; the requested device must be available |
| `validation.seed` | `2026` |
| `validation.threads` | `4` |
| `validation.quick_timeout_s` | `900` s per quick, numerical, dataset-check, or aggregate-render command |
| `validation.generation_timeout_s` | `1800` s |
| `validation.training_timeout_s` | `7200` s for each train/build/predict command |
| Foreground waiting and automatic fetch | On |
| Dataset cache reuse | On, with hashes and validation checked |
| Dry run | Off |
| `validate` suite | `smoke` |

The TOML `schema_version` is `1`. Remote paths and the interpreter are listed in
the [frozen environment guide](remote-environment.md#configured-paths-and-runtime).
Worker commands use separate process groups; a timeout stops the group and
force-kills remaining processes after a five-second grace period.

## Dataset

The bundled interval covers the complete 4500 s S5 VLA recording: all 21
channels contain 6,750,000 samples at 1500 Hz. The end value is the last
window's **start time**, not its end time. Disjoint 1 s windows produce 4500
samples per route, with 3600 train and 900 test rows under the full_interleaved 4:1
split. Neural training reserves 25% of the train rows for validation when no
validation files exist, leaving 2700 optimizer-training and 900 validation rows.
This is a within-record baseline; it does not measure independent time/track
generalization.

| Field in `[dataset]` | Default |
|---|---|
| `event_name` | `S5` |
| `fs` | `1500` Hz |
| `segment_start_s` | `0` s |
| `segment_end_s` | `4499` s; window starts include both endpoints |
| `segment_duration_s` | `1.0` s |
| `segment_step_s` | `1.0` s |
| `num_snapshots_per_segment` | `1` |
| `snapshot_overlap_count` | `0` |
| `frequency_selection_modes` | `["deep", "shallow"]` |
| `split_strategy` | `full_interleaved` |
| `train_test_ratio` | `[4, 1]`, train:test |
| `train_side` | `before`; used only by `range_overlap` |
| `half_duration_s` | `800` s per candidate side; used only by `range_overlap` |
| `gap_s` | `15` s; used only by `range_overlap` |
| `test_step_s` | `1` s; used only by `range_overlap` |
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
| `epochs` | `100` |
| `batch_size` | `64`, for both train and predict |
| `base_channels` | `16` for CNNs; ResNet uses its architecture's channel layout |
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
workflow TOML fields or `run.sh` options. Manual Python uses train batch size 16,
prediction batch size 32, and normalized training loss; see
[training and prediction](training-and-prediction.md#train-options).

## GRNN

GRNN builds a fresh reference and uses it explicitly for prediction. It has
no epochs or optimizer and uses `validation.seed` for cross-validation.
The same settings apply independently to RBD, ELM, and SCM; references and
bandwidth search results are kept under their respective method directories.

| Field in `[grnn]` | Default |
|---|---|
| `cv_folds` | `5` |
| `sigma_candidates` | `0.01`–`0.10` in steps of `0.01`, then `0.20`–`2.00` in steps of `0.10` (29 candidates) |
| `batch_size` | `1024`, for both build and predict |

Feature-coordinate standardization and per-sample channel normalization are
both off. The direct GRNN CLI defaults to no CV search and batch size 128; see
[manual build options](standalone-grnn.md#build-options).

## A5000 Sizing

The defaults target haiqin's RTX A5000 (24 GB), i7-11700K (8 cores/16 threads),
and 30 GiB RAM. Profile `20261008T163807Z-4786a709` used the frozen environment,
real S5 HDF5 data, current float32 training, AdamW, and the shared epoch loop.
Each neural setting had three warm-up batches and two measured 2700-row train
epochs; the table shows their mean time and peak PyTorch reserved GPU memory.

| Batch size, 4 CPU threads | SCM ResNet50 epoch | SCM ResNet50 reserved memory | ELM complex CNN epoch |
|---|---|---|---|
| 16 | 7.608 s | 1.506 GiB | 2.114 s |
| 32 | 6.689 s | 2.588 GiB | 1.825 s |
| 64 | 6.448 s | 5.012 GiB | 1.684 s |
| 128 | 6.293 s | 9.889 GiB | 1.638 s |

Batch 64 retains about 97–98% of the measured batch-128 throughput and gives
43 optimizer steps per epoch. It leaves room for CUDA/context overhead and
other applications. Four CPU threads remain the baseline: eight did not give
a consistent improvement in these measurements. CNN base channels remain 16,
matching the model defaults; model width and epoch count are study settings.

The historical SCM-GRNN prediction measurement used all 3600 reference and 900 test rows, with one warm-up
and five measured predictions per setting. Mean prediction times for batches
128/256/512/1024 were 23.17/18.15/15.56/14.58 ms. Batch 1024 peaked at 414.4 MiB
of allocated GPU memory and covers each 720-row CV fold or the 900-row test set
in one block. These are prediction timings, not total reference-build timings.

The profile measures runtime and memory; it does not establish an optimal
batch size for accuracy or convergence. Raw observations and the profiling
script are in the profile's local `output/data/<run_id>/` directory.

## Result Delivery

| Output | Default |
|---|---|
| Logs, tables, and reports | Local `output/data/<run_id>/` |
| Figures | Local `output/figure/<run_id>/` |
| Figure format | PDF only; embedded raster content uses 300 dpi |
| Original per-model plots | PDF only |
| HDF5 datasets and `.pt` artifacts | Kept on haiqin |

See [result visualization](result-visualization.md) for the directory layout
and figure definitions.
