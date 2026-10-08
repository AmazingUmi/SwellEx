# Training and Prediction

RBD, ELM, and SCM use independent loaders and model registries with shared
training and prediction helpers. For HDF5 layouts and pair ordering, see the
[dataset contract](../datasets/hdf5-datasets-and-splits.md). GRNN uses a
[separate build/predict workflow](standalone-scm-grnn.md).

## Method and Model Selection

For a complete selected-model run, dispatch from the local project root:

```bash
./scripts/run.sh SCM ResNet18 --epochs 100 --seed 2026
```

The public architectures are exactly `complex_CNN`, `real_CNN`, `ResNet18`,
and `ResNet50` for each method, plus `SCM GRNN`. This entry freezes configuration,
validates or reuses all three aligned feature datasets, executes only the chosen
model, renders figures, and retrieves results. See the
[remote workflow](remote-validation.md#select-one-model) for configuration,
cache policy, detached execution, and engineering validation boundaries.
Selected neural runs start fresh; continuation remains a manual operation.
The [command defaults](remote-validation.md#command-defaults) list the effective
shell settings and their sources, including two epochs, batch size 8 for train
and predict, and km-space loss.

The shell command maps the public architecture to the method's registered
model name. Manual Python commands use the registered names below:

| METHOD | Entry point | Model naming |
|---|---|---|
| RBD | `scripts/localization_networks/RBD_method/Network_main.py` | Base architecture name, without a method prefix |
| ELM | `scripts/localization_networks/ELM_method/Network_main.py` | Base architecture name with the `elm_` prefix |
| SCM | `scripts/localization_networks/SCM_method/Network_main.py` | Base architecture name with the `scm_` prefix |

The [model architecture guide](../methods/trainable-model-architecture.md)
lists the full registered names. Each method's `network/model.py:MODEL_REGISTRY`
defines its available models.

## Train and Predict on haiqin

Use the interpreter and frozen snapshot prepared by the
[remote workflow](remote-validation.md#manual-commands-in-a-frozen-run).
The following RBD example uses its `SWELLEX_PYTHON` variable from the run's
`project/` directory; substitute the entry point, model, and dataset together
for ELM or SCM.

```bash
"$SWELLEX_PYTHON" -B scripts/localization_networks/RBD_method/Network_main.py train \
  --model complex_cnn_range --data <rbd_dataset_name> --device cuda \
  --no-resume --epochs 100 --loss-space km --huber-beta 0.5

"$SWELLEX_PYTHON" -B scripts/localization_networks/RBD_method/Network_main.py predict \
  --model complex_cnn_range --data <rbd_dataset_name> --device cuda
```

Dataset names identify directories under remote `outputs/Datasets/` or an
existing dataset path. Prediction reads `*_test.h5`; it does not accept an
`--input-data` override. The generation report provides the route-to-dataset
mapping; choose a model from that method's registry and the architecture guide.
Selected runs and the automatic smoke/experiment suites perform these operations and record
commands and results; see the remote workflow for dispatch and retrieval.

## Shared Options

This table describes direct `Network_main.py` commands. The automatic shell
workflow supplies its [configured defaults](remote-validation.md#command-defaults)
for epochs, batch size, channels, loss, validation fraction, seed, and device.

| Option | Applies to | Behavior |
|---|---|---|
| `--data`, `--model` | Both | Required dataset and method-specific registered model |
| `--device cuda` | Both | Explicit GPU execution; standalone CLI otherwise chooses CUDA if available, then CPU |
| `--output-dir PATH` | Both | Base output root; adds `<model>/` unless PATH already names that model |
| `--batch-size N` | Both | Defaults: train 16, predict 32 |
| `--num-workers N` | Both | DataLoader workers; default 0 |
| `--train-data GLOB ...`, `--val-data GLOB ...` | Train | Explicit HDF5 paths/globs; quote globs |
| `--epochs N` | Train | Epoch count for a fresh run; additional epochs when resuming |
| `--lr`, `--weight-decay` | Train | Defaults `3e-4`, `1e-4` |
| `--base-channels`, `--dropout` | Train | Fresh model defaults 16 and 0.15 |
| `--loss-space normalized\|km`, `--huber-beta` | Train | SmoothL1 space; beta defaults to 0.5 in that space |
| `--val-fraction`, `--seed` | Train | Random validation fraction 0.25 and seed 2026 when no validation files exist |
| `--no-input-norm` | Train | Disable per-sample channel normalization; real CNN defaults to no input normalization |
| `--no-resume`, `--resume-checkpoint PATH` | Train | Fresh run or explicit continuation |
| `--checkpoint PATH` | Predict | Override checkpoint lookup |
| `--predictions-csv PATH`, `--plot-path PATH`, `--no-plot` | Predict | Override timestamped exports or omit the per-model plot |
| `--print-first N` | Predict | Print the first N prediction rows; default 10 |

Training first uses explicit validation files, then dataset `*_val.h5`, then a
seeded split of the training files. Target normalization fits training labels;
reported MAE/RMSE remain in km. `normalized` loss uses normalized targets;
`km` loss and its Huber beta use physical range units. SmoothL1 is not squared
range error.

Without `--no-resume`, training tries `<dataset>_last.pt`, then the most
recently modified timestamped `last` checkpoint. Resume restores model,
normalization, optimizer/scheduler, and history; dataset and input-shape
mismatches are rejected. The data split is rebuilt from the current paths,
seed, and validation options, so retain these settings for continuation;
saved split metadata is not a check that the current split is unchanged.
Prediction uses `<dataset>_best.pt`, then its
most recently modified timestamped match. A missing explicit `--checkpoint`
also triggers matching timestamped lookup. Checkpoint model and dataset names
must match the requested values.

## Remote Artifacts and Local Delivery

Default remote root: `outputs/networks_results/<METHOD>_method/<model>/`.
The automatic worker overrides it with
`outputs/networks_results/<METHOD>/<model>/` for its frozen run.
All exported basenames append `MMDD_HHMMSS` before the extension.

| Subdirectory | Basename | Contents |
|---|---|---|
| `train_outputs/` | `<dataset>_best_*.pt`, `<dataset>_last_*.pt` | Model/config, weights, optimizer/scheduler, normalization, loss space, split/source metadata, history |
| `train_outputs/` | `<dataset>_history_*.json` | Per-epoch train/validation loss, MAE, RMSE and loss space |
| `test_outputs/` | `<dataset>_predictions_*.csv` | Valid test windows and physical range errors |
| `test_outputs/` | `<dataset>_range_prediction_*.png` | Optional per-model label/prediction plot |

CSV fields are `sample`, `source_segment_idx`, `window_center_s`,
`pred_range_km`, `true_range_km`, and `abs_error_km`. Time is in seconds and
range/error in km. HDF5 and checkpoints stay on haiqin. Automatic retrieval
places logs, reports, history, and CSVs in `output/data/<run_id>/`, and figures
in `output/figure/<run_id>/`. For aggregate figure definitions and provenance,
see [result visualization](result-visualization.md).
