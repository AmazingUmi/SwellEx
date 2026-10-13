# RBD/ELM/SCM GRNN Build and Prediction

This page documents the **base-pack** GRNN CLI and explicit local CPU workflow.
GRNN stores reference features/labels and predicts by Gaussian-kernel averaging;
it has no epochs or optimizer. Equations are in the [method note](../methods/grnn.md).

For automatic remote execution:

```bash
./scripts/run.sh base_pack --method SCM --network GRNN
```

Select `RBD`, `ELM`, or `SCM`. Shell options and bundled defaults are maintained
in the [command guide](remote-validation.md) and [configuration table](command-defaults.md#grnn).

## Manual Build and Predict

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run)
on haiqin with base-pack source paths. Use its `SWELLEX_PYTHON` interpreter
from the run's `project/` directory.

1. Choose a method and its path from `dataset_directories` in the run's `report.json`.
2. Replace `<dataset_name>` below and build the reference.
3. Copy the reference path printed by build and use it for prediction.

```bash
GRNN_METHOD='RBD'
GRNN_DATA='<dataset_name>'
GRNN_OUTPUT='outputs/manual_results/grnn'
"$SWELLEX_PYTHON" -B scripts/base_pack/localization_networks/grnn/main.py build \
  --method "$GRNN_METHOD" --data "$GRNN_DATA" --device cuda --cv-folds 3 \
  --output-dir "$GRNN_OUTPUT" \
  --sigma-candidates 0.01 0.1 1.0 --batch-size 64 --seed 2026

REFERENCE='<path_to_reference>'
"$SWELLEX_PYTHON" -B scripts/base_pack/localization_networks/grnn/main.py predict \
  --method "$GRNN_METHOD" --data "$GRNN_DATA" --device cuda --reference "$REFERENCE" --batch-size 64 \
  --output-dir "$GRNN_OUTPUT"
```

This example uses a smaller search than the workflow. Use a new output
directory to avoid ambiguous result discovery; retrieve manual results separately.

## Options Used by Both Manual Commands

These defaults belong to direct `grnn/main.py build|predict`, not `run.sh`.

| Option | Default | Meaning |
|---|---|---|
| `--method METHOD` | Required | `RBD`, `ELM`, or `SCM`; saved references and input signatures must match |
| `--data DATASET` | Required | Selected method's dataset name under `outputs/Datasets/` or an existing relative/absolute directory path |
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

Without `--reference`, lookup tries `<dataset>_reference.pt`, then the matching
`MMDD_HHMMSS` file with latest mtime. A missing explicit path fails.
References bind the method, preprocessing, shape, and physical axes. Validation
and prediction inputs must match; missing contract fields require rebuilding.
See the [dataset contract](../datasets/hdf5-datasets-and-splits.md).

## Artifacts

Default roots for manual and automatic runs are method-specific:

| Method | Model/output directory | Flattened feature dimension |
|---|---|---|
| RBD | `outputs/networks_results/RBD/rbd_grnn_range/` | `2*E*F` |
| ELM | `outputs/networks_results/ELM/elm_grnn_range/` | `E*(E-1)*F` |
| SCM | `outputs/networks_results/SCM/scm_grnn_range/` | `E*(E+1)*F` |

`--output-dir` defaults to `outputs/networks_results/<METHOD>/`; the command
appends the model directory unless it is already the supplied directory name.
References bind the method, frequency/array/pair axes, and preprocessing;
equal vector lengths do not permit reuse between methods.

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

## Local CPU Range-Overlap Computation

Use this only when local execution is requested. The base worker runs one
`range_overlap` direction without SSH: generate/reuse one method, select sigma
with training-side CV, build on all training rows, predict the opposite side,
and deliver results.

| Option | Default | Meaning / constraint |
|---|---|---|
| `--method METHOD` | Required | `RBD`, `ELM`, or `SCM` |
| `--python PATH` | Required | Existing CPU environment interpreter |
| `--preset` | `full` (rejected here) | Choose `train_before` or `train_after` |
| `--config PATH` | — | Complete range-overlap TOML; exclusive with `--preset` |
| `--data PATH` | Generate fresh | Reuse a matching method dataset after settings/split/raw-input validation |

The project lock is for Linux/CUDA. For macOS CPU execution, use a separate
Python 3.14 environment with the same direct numerical dependency versions:

```bash
UV_CACHE_DIR=/private/tmp/swellex-grnn-uv-cache \
  uv venv --python python3.14 /private/tmp/swellex-grnn-local-env
UV_CACHE_DIR=/private/tmp/swellex-grnn-uv-cache \
  uv pip install --python /private/tmp/swellex-grnn-local-env/bin/python \
  --no-config --index-url https://pypi.org/simple \
  numpy==2.4.3 scipy==1.18.1 h5py==3.16.0 torch==2.11.0 matplotlib==3.10.9

./scripts/run.sh base_pack local-grnn --method RBD \
  --preset train_before \
  --python /private/tmp/swellex-grnn-local-env/bin/python
```

This command computes immediately in the local foreground; interruption stops it.
The worker explicitly uses CPU and configured threads; remote host/device and neural fields are unused. It does not install
packages. Set `MPLCONFIGDIR` to a writable directory if needed.

Reuse checks method/configuration, split membership, disjoint sample intervals,
range tracks, and raw-input path/size/mtime/SHA256. Opposite-side or changed
inputs fail. Fresh generation uses a unique variant tag to preserve datasets.

| Location | Saved evidence |
|---|---|
| `output/data/<run_id>/` | Report, config, commands/logs, complete Python snapshot, package versions, reference, search JSON, prediction CSV, hashes |
| `output/figure/<run_id>/` | Prediction and sigma-search PDFs |
| `outputs/Datasets/` | Method HDF5 and generation metadata |

Prediction explicitly uses the current run's reference. Final checks reject
source/input changes during execution. Random CV, both-sides range selection,
and full-record mean centering retain the [method's evaluation boundaries](../methods/grnn.md#sigma-选择与评估边界).
