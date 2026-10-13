# Result Figures and Manual Plotting

Selected-model, smoke, and experiment runs automatically render figures after
successful model evaluation. Foreground runs retrieve them on completion;
after stopping local monitoring, use [status and fetch](remote-validation.md#task-id-status-and-fetch).

## Find Delivered Results

```text
output/data/<run_id>/
  report.json                       final run status
  manifest.json, request.json       source identity and frozen settings
  environment.json                  actual runtime
  delivery-verification.json        retrieved visualization-file verification
  logs/                             commands and train/test logs
  reports/                          checks and metrics
  reports/visualization/            metrics.csv, source-report.json, manifest.json
  project/outputs/networks_results/  prediction CSVs and history JSONs
output/figure/<run_id>/
  range-time.pdf
  prediction-scatter.pdf
  residuals-time.pdf
  absolute-error-ecdf.pdf
  error-metrics.pdf
  models/<METHOD>/<model>/
    training/history.pdf            neural training curves
    test_outputs/prediction.pdf      per-model test predictions
  grnn/<method-model>-sigma.pdf
```

HDF5 datasets, checkpoints, and GRNN references remain on haiqin in the run's
`project/outputs/`. Do not commit `output/` or `outputs/`.

## Choose a Figure

| Question | Figure | Units or interpretation |
|---|---|---|
| How does predicted range follow time? | `range-time` | Every label/prediction point; window center in s, range in km |
| How close are predictions to labels? | `prediction-scatter` | Shared x/y scales and ideal `y=x` line |
| When are predictions too high or too low? | `residuals-time` | Prediction minus label in km; symmetric limits |
| What fraction of windows has error below a value? | `absolute-error-ecdf` | All absolute errors sorted; cumulative rank divided by n |
| How large are the errors overall? | `error-metrics` | MAE, RMSE, and maximum absolute error in km; axes start at zero |
| How did neural training evolve? | `models/<METHOD>/<model>/training/history.pdf` | Original train/validation loss, MAE, and RMSE by epoch |
| Which GRNN sigma was selected? | `grnn/` | CV mean or held-out metrics; minimum-MAPE sigma marked |

Base grids use RBD/ELM/SCM rows and shared architecture columns; joint grids
use feature-combination rows, and joint ECDFs separate architectures. Display
selection omits unused panels while preserving comparable scales.

All figures are PDF and retain every valid test row. Time plots use points;
there is no smoothing, downsampling, pooling of runs, or invented uncertainty.
The renderer reads recorded CSV/history; GRNN has a search curve when available
and no epoch history. SmoothL1 labels retain the configured loss space.

## Render Manually on haiqin

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run).
`SWELLEX_RUN` must point to the remote run directory containing `report.json`,
and `SWELLEX_PYTHON` must be its locked interpreter.

The examples below use a base-pack run. For a joint run, use
`scripts/joint_feature_pack/visualization/plot_results.py` and its registered
model names. From the frozen `project/` directory, render to a new directory:

```bash
"$SWELLEX_PYTHON" -B scripts/base_pack/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/manual-all
```

To compare specific models from a completed **selected, smoke, or experiment** run,
choose another new directory and repeat `--model`:

```bash
"$SWELLEX_PYTHON" -B scripts/base_pack/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/manual-comparison \
  --model RBD/complex_cnn_range --model SCM/scm_grnn_range
```

A filtered run contains the complete method/network selection. Existing output
directories are rejected. Manual rendering writes figures and CSV/JSON provenance together;
these new files require separate retrieval and are not added to a completed
run's automatic export.

| Option | Default | Meaning |
|---|---|---|
| `--run PATH` | Required | One remote run directory with a PASS selected/smoke/experiment report |
| `--output-dir PATH` | Snapshot project's `output/figure/<run_id>/` | New rendering directory; explicit relative paths use the current working directory |
| `--model METHOD/MODEL` | All report models | Select a registered model from the report; repeat for multiple models |
| `--dpi N` | `300` | Resolution for any embedded raster content in PDF, integer from 72 to 1200 |

`quick` runs test plotting with synthetic results; they do not produce
scientific result figures.

## Input Checks and Provenance

| Check | Required condition |
|---|---|
| Plan | Complete recorded suite: base smoke/full 6/15; joint smoke/full 14/35; selected: complete filtered plan |
| Evaluation | Every planned model passed evaluation |
| Files | Exactly one prediction CSV per model and one history per neural model |
| Values | Finite rows, consistent errors/metrics, and recorded loss space |
| Identity | Matching window indices, times, and true labels across models; canonical method/network filters |
| Joint run | Matching pack/fusion and complete joint plan |

Checks cover the entire run before `--model` filtering. Discovery is limited
to `project/outputs/networks_results/`, excluding resume-check histories and
other runs. Base historical 4/13-model reports require their known SCM-only
source-plan identity; this exception does not relax new plan completeness.

| File or field | Meaning |
|---|---|
| Local `report.json` | Authoritative final run status |
| Local `delivery-verification.json` | Terminal fetch's size/hash verification of visualization manifest files; present when that manifest exists |
| `source-report.json` | Report captured when rendering began; an automatic run may still show RUNNING here |
| Visualization `manifest.json` | Input/export hashes, actual plotting package versions, counts, and transformations |
| `delivery_roots` | Automatic local data/figure roots, relative to the local project |
| File `group` and `path` | `data` or `figure`, with a path relative to that delivery root |
| `render_output_dir` | Actual rendering directory on haiqin |
| `purpose` | `engineering_validation` for smoke; no-accuracy-threshold purpose for selected/experiment runs |

Fetch verifies safe relative paths, sizes, and SHA256 for manifest-listed PDFs,
metrics, and captured source reports. Other logs use rsync checksum retrieval
and are outside that manifest's scope. Renderer source hashes are separate
from training source hashes.

A PASS model run without a rendering manifest fails delivery verification;
remote computation status remains separately recorded in `report.json`.
Pipeline PASS carries no scientific accuracy acceptance threshold.
