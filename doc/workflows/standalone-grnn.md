# RBD/ELM/SCM GRNN Build and Prediction

GRNN stores the selected RBD, ELM, or SCM features and range labels as reference samples, then
predicts by Gaussian-kernel averaging. It uses no epochs, optimizer, or
backpropagation. See the [method note](../methods/grnn.md) for the equations.

## Run the Complete Workflow

From the local project root:

```bash
./scripts/run.sh RBD GRNN
./scripts/run.sh ELM GRNN
./scripts/run.sh SCM GRNN
```

The command prepares and validates the three feature datasets, builds a fresh
method-specific reference, selects sigma independently for that method, predicts,
plots, and retrieves results. The complete experiment has 15 models; smoke has
three complex CNNs and three GRNNs. GRNN stays outside neural model registries.
Bundled GRNN settings are 5-fold cross-validation, 29 candidates (0.01–0.10 in
steps of 0.01, then 0.20–2.00 in steps of 0.10), batch size 1024, and seed 2026.

Use `--seed N` or `--config PATH` to change run settings. `--epochs` is rejected.
See the [shell guide](remote-validation.md) for background runs and retrieval,
and [command defaults](command-defaults.md#grnn) for the full configuration.

## Manual Build and Predict

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run)
on haiqin. Choose a run from the three-method GRNN version; older SCM-only
snapshots retain their historical CLI. Use its `SWELLEX_PYTHON` interpreter from
the run's `project/` directory.

1. Choose a method and its path from `dataset_directories` in the run's `report.json`.
2. Replace `<dataset_name>` below and build the reference.
3. Copy the reference path printed by build and use it for prediction.

```bash
GRNN_METHOD='RBD'
GRNN_DATA='<dataset_name>'
GRNN_OUTPUT='outputs/manual_results/grnn'
"$SWELLEX_PYTHON" -B scripts/localization_networks/grnn/main.py build \
  --method "$GRNN_METHOD" --data "$GRNN_DATA" --device cuda --cv-folds 3 \
  --output-dir "$GRNN_OUTPUT" \
  --sigma-candidates 0.01 0.1 1.0 --batch-size 64 --seed 2026

REFERENCE='<path_to_reference>'
"$SWELLEX_PYTHON" -B scripts/localization_networks/grnn/main.py predict \
  --method "$GRNN_METHOD" --data "$GRNN_DATA" --device cuda --reference "$REFERENCE" --batch-size 64 \
  --output-dir "$GRNN_OUTPUT"
```

The example uses a smaller search than the bundled workflow. The direct CLI
defaults below differ. Manual results need separate retrieval and do not
update a completed run's report.

Choose a new `GRNN_OUTPUT` directory for each manual run. Adding result files
to the original automatic output directory would make its prediction/search
files ambiguous when rendering again.

## Options Used by Both Manual Commands

The sole direct entry is `scripts/localization_networks/grnn/main.py`.
These tables describe its `build` and `predict` commands, not `run.sh`.

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

Without `--reference`, prediction tries `<dataset>_reference.pt`, then the
matching `MMDD_HHMMSS` reference with latest mtime. Prediction requires the
saved `[2,E|P,F]` shape and `input_signature` to match the input data exactly,
including frequency, array, and pair axes. Explicit validation files must match
the reference-building inputs by the same rule. A reference with missing
contract fields, including older files without `input_signature`, is rejected;
rebuild it with the current entry. The
[dataset contract](../datasets/hdf5-datasets-and-splits.md) defines the required
metadata and axis handling.

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

When local execution is requested, `workflow/local_grnn.py` runs the complete
selected-method GRNN calculation through `./scripts/run.sh local-grnn`, without SSH.
It reuses the existing before/after TOML
presets, generates only `--method`, selects sigma by training-side random CV, builds
the reference from all training rows, and predicts every opposite-side test
row. Each direction gets its own reference and run directory.

The project lock is for Linux/CUDA. For macOS CPU execution, use a separate
Python 3.14 environment with the same direct numerical dependency versions:

```bash
UV_CACHE_DIR=/private/tmp/swellex-grnn-uv-cache \
  uv venv --python python3.14 /private/tmp/swellex-grnn-local-env
UV_CACHE_DIR=/private/tmp/swellex-grnn-uv-cache \
  uv pip install --python /private/tmp/swellex-grnn-local-env/bin/python \
  --no-config --index-url https://pypi.org/simple \
  numpy==2.4.3 scipy==1.18.1 h5py==3.16.0 torch==2.11.0 matplotlib==3.10.9

./scripts/run.sh local-grnn --method RBD \
  --config scripts/localization_networks/workflow/haiqin_range_overlap_train_before.toml \
  --python /private/tmp/swellex-grnn-local-env/bin/python --dry-run

./scripts/run.sh local-grnn --method ELM \
  --config scripts/localization_networks/workflow/haiqin_range_overlap_train_after.toml \
  --python /private/tmp/swellex-grnn-local-env/bin/python --dry-run
```

`--dry-run` uses only the standard-library dispatcher and prints the planned
command; it does not import numerical packages, create a run, or contact haiqin.
Remove it when local computation is intended. `--method` and `--python` are required; the latter selects
the already-installed CPU environment; set `MPLCONFIGDIR` to a writable directory
when needed. The workflow does not create or modify the numerical environment.

The entry explicitly uses CPU and the configured thread count; the TOML's
remote device/host and neural training fields are unused. It prints `Run:`
before computation, rejects `full_interleaved`, and gives generation a unique
variant tag so existing datasets are preserved. Optional `--data PATH` reuses
one matching method directory after checking its method and generation settings, complete
split membership, disjoint original sample intervals, and identical raw range
tracks. It also requires metadata's complete `input_files` inventory and verifies
the recorded raw files against their size, modification time, and SHA256; missing
or changed input identity is rejected. An opposite-side dataset is rejected.

`output/data/<run_id>/` keeps the report, configuration, executed build/predict
arguments, logs, complete Python source snapshot including method loaders,
package-version freeze, reference, sigma-search
JSON, and prediction CSV. `output/figure/<run_id>/` keeps the prediction and
sigma-search PDFs; method HDF5 and metadata remain under `outputs/Datasets/`.
Reports record input/source hashes, all split identities, selected sigma, and
delivered-CSV MAE/RMSE/MAPE. Local references also record training-side window
identities and the training HDF5 hash. Prediction always uses this run's explicit
reference. Final checks reject source/input changes during computation.

Sigma search reuses each fold's batched distances across candidates. Each fold
still fits preprocessing only on its training rows; the kernel and minimum-MAPE
selection rule are unchanged. The test side never selects sigma. Random CV,
the range interval selected from both sides' true labels, and full-record mean
centering retain the [method's evaluation boundaries](../methods/grnn.md).
