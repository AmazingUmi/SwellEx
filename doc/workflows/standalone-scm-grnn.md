# SCM-GRNN Build and Prediction

SCM-GRNN stores SCM features and range labels as reference samples, then
predicts by Gaussian-kernel averaging. It uses no epochs, optimizer, or
backpropagation. See the [method note](../methods/scm-grnn.md) for the equations.

## Run the Complete Workflow

From the local project root:

```bash
./scripts/run.sh SCM GRNN
```

The command prepares and validates the three feature datasets, builds a fresh
SCM reference, selects sigma, predicts, plots, and retrieves results.
Bundled GRNN settings are 3-fold cross-validation, candidates `[0.01, 0.1, 1.0]`,
batch size 64, and seed 2026.

Use `--seed N` or `--config PATH` to change run settings. `--epochs` is rejected.
See the [shell guide](remote-validation.md) for background runs and retrieval,
and [command defaults](command-defaults.md#scm-grnn) for the full configuration.

## Manual Build and Predict

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run)
on haiqin. Use its `SWELLEX_PYTHON` interpreter from the run's `project/` directory.

1. Choose an SCM path from `dataset_directories.SCM` in the run's `report.json`.
2. Replace `<scm_dataset_name>` below and build the reference.
3. Copy the reference path printed by build and use it for prediction.

```bash
SCM_DATA='<scm_dataset_name>'
GRNN_OUTPUT='outputs/manual_results/grnn'
"$SWELLEX_PYTHON" -B scripts/localization_networks/grnn/main.py build \
  --data "$SCM_DATA" --device cuda --cv-folds 3 \
  --output-dir "$GRNN_OUTPUT" \
  --sigma-candidates 0.01 0.1 1.0 --batch-size 64 --seed 2026

REFERENCE='<path_to_reference>'
"$SWELLEX_PYTHON" -B scripts/localization_networks/grnn/main.py predict \
  --data "$SCM_DATA" --device cuda --reference "$REFERENCE" --batch-size 64 \
  --output-dir "$GRNN_OUTPUT"
```

The example explicitly uses the shell workflow's search settings. The direct
CLI defaults below differ. Manual results need separate retrieval and do not
update a completed run's report.

Choose a new `GRNN_OUTPUT` directory for each manual run. Adding result files
to the original automatic output directory would make its prediction/search
files ambiguous when rendering again.

## Options Used by Both Manual Commands

The sole direct entry is `scripts/localization_networks/grnn/main.py`.
These tables describe its `build` and `predict` commands, not `run.sh`.

| Option | Default | Meaning |
|---|---|---|
| `--data DATASET` | Required | SCM dataset name under `outputs/Datasets/` or an existing relative/absolute directory path |
| `--device DEVICE` | CUDA if available, otherwise CPU | Supply `cuda` explicitly for remote execution |
| `--output-dir PATH` | GRNN output root below | Root for reference and prediction files |
| `--batch-size N` | `128` | Batch size for distance-based prediction |

## Build Options

| Option | Default | Meaning |
|---|---|---|
| `--train-data GLOB ...` | Dataset `*_train.h5` | Reference files; quote globs |
| `--val-data GLOB ...` | None | Explicit held-out validation files; quote globs |
| `--spread VALUE` | `0.01` | Fixed sigma when no search is selected |
| `--cv-folds N` | `0` | Search with reference-set k-fold CV when N > 1 |
| `--seed N` | `2026` | Cross-validation seed |
| `--sigma-candidates VALUE ...` | Candidate grid below | Sigma values evaluated during search |
| `--standardize-input` | Off | Fit feature-coordinate z-scores on reference samples |
| `--per-sample-channel-norm` | Off | Normalize real/imag channels before flattening |
| `--max-reference-samples N` | No limit | Limit loaded reference samples for debugging |
| `--max-validation-samples N` | No limit | Limit loaded validation samples for debugging |

The manual default candidate grid is 0.01–0.10 in steps of 0.01, followed by
0.20–2.00 in steps of 0.10. Spread and candidates must produce a finite positive
normal float32 kernel denominator `2*sigma^2`; extreme magnitudes are rejected.
Batch size and sample limits must be positive integers. `--cv-folds` accepts
0 or an integer of at least 2. Seeds must be in `0..2**32-1`; abbreviated
option names are rejected.

### How Sigma Is Chosen

The first applicable rule wins:

1. Explicit `--val-data`: search candidates and minimize held-out MAPE.
2. `--cv-folds` greater than 1: search candidates and minimize mean fold MAPE.
3. Otherwise: use the fixed `--spread` value.

Candidates alone do not start a search. GRNN does not automatically use
`*_val.h5`; pass `--val-data` for that mode. Normalization and standardization
settings are saved in the reference and reused during prediction.

## Predict Options

| Option | Default | Meaning |
|---|---|---|
| `--reference PATH` | Automatic reference lookup | Use a specific `.pt` reference; a missing explicit file fails |
| `--input-data GLOB ...` | Dataset `*_test.h5` | Explicit test files; quote globs |
| `--max-predict-samples N` | No limit | Limit loaded test samples for debugging |
| `--predictions-csv PATH` | Timestamped CSV | Set the prediction CSV path |
| `--plot-path PATH` | Timestamped PDF | Set the per-model plot path; must end in `.pdf` |
| `--no-plot` | Off | Omit the per-model plot |
| `--print-first N` | `10` | Number of prediction rows printed |

Without `--reference`, prediction tries `<dataset>_reference.pt`, then the
matching `MMDD_HHMMSS` reference with latest mtime. Prediction requires the
saved `[2,P,F]` shape and `input_signature` to match the input data exactly,
including frequency, array, and pair axes. Explicit validation files must match
the reference-building inputs by the same rule. A reference with missing
contract fields, including older files without `input_signature`, is rejected;
rebuild it with the current entry. The SCM
[dataset contract](../datasets/hdf5-datasets-and-splits.md) defines the required
metadata and axis handling.

## Artifacts

Default root for both manual and automatic runs:
`outputs/networks_results/SCM/scm_grnn_range/`.
`--output-dir` defaults to `outputs/networks_results/SCM/`; the command appends
`scm_grnn_range/` unless that is already the supplied directory name.

| Subdirectory | File | Contents |
|---|---|---|
| `reference_outputs/` | `<dataset>_reference_MMDD_HHMMSS.pt` | Reference features/labels, shape, input signature, normalization, selected sigma, and search metadata |
| `reference_outputs/` | `<dataset>_sigma_search_MMDD_HHMMSS.json` | Candidate metrics when a search was performed |
| `test_outputs/` | `<dataset>_predictions_MMDD_HHMMSS.csv` | [Shared prediction CSV fields](training-and-prediction.md#remote-artifacts-and-local-delivery) |
| `test_outputs/` | `<dataset>_range_prediction_MMDD_HHMMSS.pdf` | Optional PDF from manual prediction; automatic plotting uses the renderer |

Automatic retrieval keeps references on haiqin, puts logs/tables in
`output/data/<run_id>/`, and puts figures in `output/figure/<run_id>/`.
GRNN has a sigma-search figure when search was performed, and no epoch history.
See [result visualization](result-visualization.md).
