# Neural Training and Prediction

This page documents direct **base-pack** Python commands. For automatic runs,
use the [shell guide](remote-validation.md); for joint inputs, use the
[joint contract](../packs/joint-feature-pack.md). GRNN has a separate
[build/predict CLI](standalone-grnn.md).

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
"$SWELLEX_PYTHON" -B scripts/base_pack/localization_networks/rbd/main.py train \
  --model complex_cnn_range --data "$RBD_DATA" --device cuda \
  --output-dir "$NEURAL_OUTPUT" \
  --no-resume --epochs 100 --loss-space km --huber-beta 0.5

CHECKPOINT='<path_to_best_checkpoint>'
"$SWELLEX_PYTHON" -B scripts/base_pack/localization_networks/rbd/main.py predict \
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
| RBD | `scripts/base_pack/localization_networks/rbd/main.py` | None |
| ELM | `scripts/base_pack/localization_networks/elm/main.py` | `elm_` |
| SCM | `scripts/base_pack/localization_networks/scm/main.py` | `scm_` |

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

These tables describe direct `main.py train` and `predict`, not `run.sh`.

| Option | Default | Meaning |
|---|---|---|
| `--data DATASET` | Required | Dataset name or directory |
| `--model MODEL` | Required | Method-specific registered name |
| `--device DEVICE` | CUDA if available, otherwise CPU | Supply `cuda` explicitly for the remote workflow |
| `--output-dir PATH` | Method's output root | Adds `<model>/` unless the path already names that model |
| `--batch-size N` | Train `16`; predict `32` | Positive samples per batch; workflow TOML defaults to `64` |
| `--num-workers N` | `0` | DataLoader workers |

## Train Options

| Option | Default | Meaning |
|---|---|---|
| `--train-data GLOB ...` | Dataset `*_train.h5` | Explicit training files; quote globs |
| `--val-data GLOB ...` | See validation order below | Explicit validation files; quote globs |
| `--epochs N` | `100` | Positive fresh epochs, or additional epochs when resuming |
| `--lr VALUE` | `3e-4` | Positive learning rate |
| `--weight-decay VALUE` | `1e-4` | Nonnegative optimizer weight decay |
| `--base-channels N` | `16` | Fresh CNN width; ResNet uses its own layout |
| `--dropout VALUE` | `0.15` | Fresh model dropout in `[0,1)` |
| `--loss-space SPACE` | `normalized` | `normalized` or `km` SmoothL1 loss |
| `--huber-beta VALUE` | `0.5` | Nonnegative SmoothL1 transition in the chosen loss space |
| `--val-fraction VALUE` | `0.25` | Internal holdout fraction in `(0,1)` when validation files are absent |
| `--seed N` | `2026` | Split seed in `0..4294967295` |
| `--no-input-norm` | Off | Disable per-sample real/imag channel normalization; real CNN already defaults to off |
| `--no-resume` | Off | Start fresh; cannot be combined with `--resume-checkpoint` |
| `--resume-checkpoint PATH` | Automatic lookup | Continue from a specific checkpoint; a missing explicit file fails |

Training chooses validation data in this order:

1. Explicit `--val-data` files.
2. Dataset `*_val.h5` files.
3. A seeded split from the training files using `--val-fraction`.

Target normalization fits training labels only. Reported MAE/RMSE are always
in km. Loss uses normalized targets for `normalized`, and physical km errors
for `km`; Huber beta follows that space. SmoothL1 is not squared range error.

Training requires nonempty training/validation sets. Nonfinite settings and
abbreviated options are rejected.

## Predict Options

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | Automatic best-checkpoint lookup | Select a checkpoint; see lookup rules below |
| `--predictions-csv PATH` | Timestamped CSV | Set the prediction CSV path |
| `--plot-path PATH` | Timestamped PDF | Set the per-model plot path; must end in `.pdf` |
| `--no-plot` | Off | Omit the per-model plot |
| `--print-first N` | `10` | Number of prediction rows printed |

## Resume and Checkpoint Lookup

| Operation | Lookup when no explicit path is supplied |
|---|---|
| Train without `--no-resume` | `<dataset>_last.pt`, then the matching timestamped `last` file with latest mtime |
| Predict | `<dataset>_best.pt`, then the matching timestamped `best` file with latest mtime |

Missing explicit train or prediction checkpoint paths fail immediately.
The timestamped fallback considers valid `MMDD_HHMMSS` suffixes and selects
by latest mtime. If automatic training lookup finds nothing, training starts fresh.

Checkpoints require the current version, method/model/dataset names, and physical
input signature. Frequency, element/depth, or pair changes fail even when shapes
match. Older artifacts without this contract must be regenerated.

Resume restores weights, model configuration, normalization, loss settings,
optimizer state, and history. Rebuilt split indices, split mode, and seed must
match the saved values. Base checks do not hash source-file contents, so keep
the files unchanged. Joint checkpoints additionally verify ordered source paths,
sizes, and SHA256 hashes, including the historical best checkpoint.

Each invocation runs a new cosine learning-rate phase from its `--lr` to zero
over its `--epochs`. Resuming restores optimizer momentum but starts this new
phase; the old scheduler cycle is not continued. History records the rate used
for each epoch (`lr`) and the following rate (`next_lr`). If no validation
improvement occurs, training retains and prints the existing best checkpoint.

## Remote Artifacts and Local Delivery

Manual and automatic default root: `outputs/networks_results/<METHOD>/<model>/`,
where `<METHOD>` is `RBD`, `ELM`, or `SCM`.
Exported basenames have an `MMDD_HHMMSS` suffix.

| Subdirectory | File | Contents |
|---|---|---|
| `train_outputs/` | `<dataset>_best_*.pt`, `<dataset>_last_*.pt` | Version, input signature, model/config, weights, optimizer, scheduler plan/state, normalization, split/source metadata, and history |
| `train_outputs/` | `<dataset>_history_*.json` | Per-epoch train/validation loss, MAE, RMSE, and loss space |
| `test_outputs/` | `<dataset>_predictions_*.csv` | Valid test-window predictions and errors |
| `test_outputs/` | `<dataset>_range_prediction_*.pdf` | Optional PDF from manual prediction; automatic plotting uses the renderer |

CSV fields are `sample`, `source_segment_idx`, `window_center_s`,
`pred_range_km`, `true_range_km`, and `abs_error_km`. Time is in seconds;
range and error are in km.

Automatic retrieval puts logs, reports, history, and CSVs in
`output/data/<run_id>/`, and figures in `output/figure/<run_id>/`.
HDF5 and checkpoints stay on haiqin. See
[result visualization](result-visualization.md) for aggregate figures and provenance.
