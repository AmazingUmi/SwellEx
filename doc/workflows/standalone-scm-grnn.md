# Standalone SCM-GRNN Workflow

GRNN builds a reference artifact from SCM features and range labels, then
predicts by Gaussian-kernel averaging. It uses no training epochs, optimizer,
or backpropagation. See the [method note](../methods/scm-grnn.md)
for its architecture and the [dataset contract](../datasets/hdf5-datasets-and-splits.md)
for SCM input layout.

## Build and Predict on haiqin

Use `SWELLEX_PYTHON` in a frozen run's `project/` directory as described by the
[remote workflow](remote-validation.md#manual-commands-in-a-frozen-run).

```bash
"$SWELLEX_PYTHON" -B scripts/localization_networks/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name> --device cuda --cv-folds 5

"$SWELLEX_PYTHON" -B scripts/localization_networks/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name> --device cuda
```

The loader reads SCM `/X` and `/y_range_km` with the same axis metadata rules
as the SCM network loader. Its flattened real/imag feature dimensions must
match the saved reference during prediction.

## Options and Sigma Selection

| Option | Applies to | Behavior |
|---|---|---|
| `--data`, `--device`, `--output-dir` | Both | Dataset, explicit device, and artifact root |
| `--batch-size N` | Both | Prediction distance batch size; default 128 |
| `--train-data GLOB ...`, `--val-data GLOB ...` | Build | Override reference/held-out validation files; quote globs |
| `--spread VALUE` | Build | Fixed sigma; default 0.01 when no search is selected |
| `--cv-folds N`, `--seed N` | Build | Reference-set k-fold search when N > 1; seed defaults to 2026 |
| `--sigma-candidates VALUE ...` | Build | Search candidates; default 0.01–0.10 in steps of 0.01, then 0.20–2.00 in steps of 0.10 |
| `--standardize-input` | Build | Fit feature-coordinate z-scores on reference samples |
| `--per-sample-channel-norm` | Build | Apply real/imag channel normalization before flattening |
| `--max-reference-samples`, `--max-validation-samples` | Build | Debugging limits on loaded samples |
| `--reference PATH`, `--input-data GLOB ...` | Predict | Override reference artifact or test files |
| `--max-predict-samples` | Predict | Debugging limit on test samples |
| `--predictions-csv`, `--plot-path`, `--no-plot`, `--print-first` | Predict | Timestamped CSV/plot overrides, optional plot, and console preview |

Sigma selection priority is explicit `--val-data`, then `--cv-folds > 1`,
then fixed `--spread`. Search minimizes held-out MAPE or mean fold MAPE.
Supplying candidates alone does not trigger a search. Normalization and
standardization settings are saved in the artifact and reused during prediction.

## Artifacts

Default remote root:
`outputs/networks_results/0_GRNN_related/scm_grnn_range/`.

| Subdirectory | Basename | Contents |
|---|---|---|
| `reference_outputs/` | `<dataset>_reference_MMDD_HHMMSS.pt` | Reference matrix/labels, input shape, normalization, selected sigma and search metadata |
| `reference_outputs/` | `<dataset>_sigma_search_MMDD_HHMMSS.json` | Candidate metrics when a search was performed |
| `test_outputs/` | `<dataset>_predictions_MMDD_HHMMSS.csv` | The [shared prediction CSV contract](training-and-prediction.md#remote-artifacts-and-local-delivery) |
| `test_outputs/` | `<dataset>_range_prediction_MMDD_HHMMSS.png` | Optional label/prediction plot |

Prediction tries the fixed `<dataset>_reference.pt`, then the most recently
modified timestamped reference, unless `--reference` is supplied. The remote
workflow keeps `.pt` artifacts on haiqin and retrieves logs/tables and figures
into the two local delivery roots. GRNN sigma-search figures are described in
[result visualization](result-visualization.md); there is no epoch history.
