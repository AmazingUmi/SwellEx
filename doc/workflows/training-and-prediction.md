# Neural Training and Prediction

RBD, ELM, and SCM each support complex CNN, real CNN, ResNet18, and ResNet50.
This guide covers complete shell runs and manual neural commands on haiqin.
SCM-GRNN uses a [separate build/predict workflow](standalone-scm-grnn.md).

## Run the Complete Workflow

From the local project root:

```bash
./scripts/run.sh SCM ResNet18 --epochs 100 --seed 2026
```

The command prepares and validates all three feature datasets, trains only
SCM ResNet18, predicts, plots, and retrieves results. It starts fresh and uses
that run's best checkpoint. See the [shell guide](remote-validation.md) for
background runs and retrieval.

Without overrides, the shell uses two epochs, train/predict batch size 8,
CNN base channels 8, and km-space loss. The complete
[default reference](command-defaults.md#neural-training-and-prediction) lists
settings and their sources. Manual Python defaults below are different.

## Manual Train and Predict

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run)
on haiqin. The commands below use its `SWELLEX_PYTHON` interpreter from the
run's `project/` directory.

1. Choose an RBD path from `dataset_directories.RBD` in the run's `report.json`.
2. Replace `<rbd_dataset_name>` below and train a fresh model.
3. Copy the best checkpoint path printed by training and use it for prediction.

```bash
RBD_DATA='<rbd_dataset_name>'
NEURAL_OUTPUT='outputs/manual_results/rbd-complex-cnn'
"$SWELLEX_PYTHON" -B scripts/localization_networks/RBD_method/Network_main.py train \
  --model complex_cnn_range --data "$RBD_DATA" --device cuda \
  --output-dir "$NEURAL_OUTPUT" \
  --no-resume --epochs 100 --loss-space km --huber-beta 0.5

CHECKPOINT='<path_to_best_checkpoint>'
"$SWELLEX_PYTHON" -B scripts/localization_networks/RBD_method/Network_main.py predict \
  --model complex_cnn_range --data "$RBD_DATA" --device cuda \
  --checkpoint "$CHECKPOINT" --output-dir "$NEURAL_OUTPUT"
```

`--data` names a directory under `outputs/Datasets/` or an existing dataset
path. Prediction reads `*_test.h5`; the neural CLI has no `--input-data` option.
Choose a new output directory for each manual run. Manual results need separate
retrieval and do not update a completed run's report.

### Choose the Method and Model

For ELM or SCM, change the entry point, dataset, and registered model together:

| Method | Python entry point | Registered-name prefix |
|---|---|---|
| RBD | `scripts/localization_networks/RBD_method/Network_main.py` | None |
| ELM | `scripts/localization_networks/ELM_method/Network_main.py` | `elm_` |
| SCM | `scripts/localization_networks/SCM_method/Network_main.py` | `scm_` |

| Shell architecture | Registered base name |
|---|---|
| `complex_CNN` | `complex_cnn_range` |
| `real_CNN` | `real_cnn_range` |
| `ResNet18` | `resnet18_range` |
| `ResNet50` | `resnet50_range` |

For example, `ELM ResNet18` maps to `elm_resnet18_range`. Each method's
`network/model.py:MODEL_REGISTRY` owns its registered names. See
[model architecture](../methods/trainable-model-architecture.md) for model details
and the [dataset contract](../datasets/hdf5-datasets-and-splits.md) for input layout.

## Options Used by Both Manual Commands

These tables describe direct `Network_main.py train` and `predict`, not `run.sh`.

| Option | Default | Meaning |
|---|---|---|
| `--data DATASET` | Required | Dataset name or directory |
| `--model MODEL` | Required | Method-specific registered name |
| `--device DEVICE` | CUDA if available, otherwise CPU | Supply `cuda` explicitly for the remote workflow |
| `--output-dir PATH` | Method's output root | Adds `<model>/` unless the path already names that model |
| `--batch-size N` | Train `16`; predict `32` | Samples per batch |
| `--num-workers N` | `0` | DataLoader workers |

## Train Options

| Option | Default | Meaning |
|---|---|---|
| `--train-data GLOB ...` | Dataset `*_train.h5` | Explicit training files; quote globs |
| `--val-data GLOB ...` | See validation order below | Explicit validation files; quote globs |
| `--epochs N` | `100` | Fresh epochs, or additional epochs when resuming |
| `--lr VALUE` | `3e-4` | Learning rate |
| `--weight-decay VALUE` | `1e-4` | Optimizer weight decay |
| `--base-channels N` | `16` | Fresh CNN width; ResNet uses its own layout |
| `--dropout VALUE` | `0.15` | Fresh model dropout |
| `--loss-space SPACE` | `normalized` | `normalized` or `km` SmoothL1 loss |
| `--huber-beta VALUE` | `0.5` | SmoothL1 transition in the chosen loss space |
| `--val-fraction VALUE` | `0.25` | Fraction held out when validation files are absent |
| `--seed N` | `2026` | Seed for the training/validation split |
| `--no-input-norm` | Off | Disable per-sample real/imag channel normalization; real CNN already defaults to off |
| `--no-resume` | Off | Start fresh even when checkpoints exist; takes precedence over an explicit resume path |
| `--resume-checkpoint PATH` | Automatic lookup | Continue from a specific checkpoint; a missing explicit file fails |

Training chooses validation data in this order:

1. Explicit `--val-data` files.
2. Dataset `*_val.h5` files.
3. A seeded split from the training files using `--val-fraction`.

Target normalization fits training labels only. Reported MAE/RMSE are always
in km. Loss uses normalized targets for `normalized`, and physical km errors
for `km`; Huber beta follows that space. SmoothL1 is not squared range error.

## Predict Options

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | Automatic best-checkpoint lookup | Select a checkpoint; see lookup rules below |
| `--predictions-csv PATH` | Timestamped CSV | Set the prediction CSV path |
| `--plot-path PATH` | Timestamped PNG | Set the per-model plot path |
| `--no-plot` | Off | Omit the per-model plot |
| `--print-first N` | `10` | Number of prediction rows printed |

## Resume and Checkpoint Lookup

| Operation | Lookup when no explicit path is supplied |
|---|---|
| Train without `--no-resume` | `<dataset>_last.pt`, then the matching timestamped `last` file with latest mtime |
| Predict | `<dataset>_best.pt`, then the matching timestamped `best` file with latest mtime |

An explicit prediction path that does not exist also triggers matching
timestamped lookup. Prediction checks that the checkpoint's model and dataset
names match the request.

Resume restores the saved model configuration, target normalization, input
normalization, loss space, optimizer/scheduler, and history. Dataset and input
shape mismatches fail. The data split is rebuilt from current files, seed, and
validation options; keep these unchanged to continue the same split. Saved
split metadata alone does not enforce that the rebuilt split is identical.

## Remote Artifacts and Local Delivery

Manual default root: `outputs/networks_results/<METHOD>_method/<model>/`.
Automatic workers use `outputs/networks_results/<METHOD>/<model>/` instead.
Exported basenames have an `MMDD_HHMMSS` suffix.

| Subdirectory | File | Contents |
|---|---|---|
| `train_outputs/` | `<dataset>_best_*.pt`, `<dataset>_last_*.pt` | Model/config, weights, optimizer/scheduler, normalization, split/source metadata, and history |
| `train_outputs/` | `<dataset>_history_*.json` | Per-epoch train/validation loss, MAE, RMSE, and loss space |
| `test_outputs/` | `<dataset>_predictions_*.csv` | Valid test-window predictions and errors |
| `test_outputs/` | `<dataset>_range_prediction_*.png` | Optional label/prediction plot |

CSV fields are `sample`, `source_segment_idx`, `window_center_s`,
`pred_range_km`, `true_range_km`, and `abs_error_km`. Time is in seconds;
range and error are in km.

Automatic retrieval puts logs, reports, history, and CSVs in
`output/data/<run_id>/`, and figures in `output/figure/<run_id>/`.
HDF5 and checkpoints stay on haiqin. See
[result visualization](result-visualization.md) for aggregate figures and provenance.
