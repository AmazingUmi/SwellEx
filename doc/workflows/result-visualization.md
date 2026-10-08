# Result Figures and Manual Plotting

Selected-model, smoke, and experiment runs automatically render figures after
successful model evaluation. Foreground runs retrieve them on completion;
for detached runs, use [status and fetch](remote-validation.md#detach-check-status-and-fetch).

## Find Delivered Results

```text
output/data/<run_id>/
  report.json                       final run status
  manifest.json, request.json       source identity and frozen settings
  environment.json                  actual runtime
  logs/                             commands and train/test logs
  reports/                          checks and metrics
  reports/visualization/            metrics.csv, source-report.json, manifest.json
  project/outputs/networks_results/  prediction CSVs and history JSONs
output/figure/<run_id>/
  range-time.png / .pdf
  prediction-scatter.png / .pdf
  residuals-time.png / .pdf
  absolute-error-ecdf.png / .pdf
  error-metrics.png / .pdf
  training/<method-model>.png / .pdf
  grnn/<method-model>-sigma.png / .pdf
  models/                           original per-model prediction PNGs
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
| How did neural training evolve? | `training/` | Original train/validation loss, MAE, and RMSE by epoch |
| Which GRNN sigma was selected? | `grnn/` | CV mean or held-out metrics; minimum-MAPE sigma marked |

A selected run has the five prediction/error figures plus its neural history
or GRNN search figure when applicable. GRNN has no epoch history. Aggregate
figures are PNG at 300 dpi and PDF; original per-model plots are PNG at 200 dpi.

Plots retain every valid test row, without smoothing, downsampling, pooling
runs, or invented error bars. Time plots use points so they do not bridge
unknown gaps. Periodic test spacing does not imply missing predictions.

SmoothL1 loss is labeled by its configured `km` or `normalized` loss space;
it is not squared range error. Figures describe the recorded run. Short smoke
runs and configured experiments do not establish scientific model superiority.

## Render Manually on haiqin

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run).
`SWELLEX_RUN` must point to the remote run directory containing `report.json`,
and `SWELLEX_PYTHON` must be its locked interpreter.

From that run's `project/` directory, render all report models to a new directory:

```bash
"$SWELLEX_PYTHON" -B scripts/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/manual-all
```

To compare specific models from a completed **smoke or experiment** run,
choose another new directory and repeat `--model`:

```bash
"$SWELLEX_PYTHON" -B scripts/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/manual-comparison \
  --model RBD/complex_cnn_range --model SCM/scm_grnn_range
```

A selected-model run contains only one model. Existing output directories are
rejected. Manual rendering writes figures and CSV/JSON provenance together;
these new files require separate retrieval and are not added to a completed
run's automatic export.

| Option | Default | Meaning |
|---|---|---|
| `--run PATH` | Required | One remote run directory with a PASS selected/smoke/experiment report |
| `--output-dir PATH` | Snapshot project's `output/figure/<run_id>/` | New rendering directory; explicit relative paths use the current working directory |
| `--model METHOD/MODEL` | All report models | Select a registered model from the report; repeat for multiple models |
| `--formats png pdf` | PNG and PDF | Choose unique formats from `png` and `pdf` |
| `--dpi N` | `300` | Export resolution, integer from 72 to 1200 |

`quick` runs test plotting with synthetic results; they do not produce
scientific result figures.

## Input Checks and Provenance

Before rendering, the public CLI checks:

- Exactly one prediction CSV per model, and one history file per neural model.
- Finite, well-formed rows and consistent recorded errors, metrics, and loss space.
- Matching source-window indices, timestamps, and true labels across compared models.
- For `selected`, exactly one model metric and a matching canonical
  `method`/`architecture`/`model` selection.

Discovery is limited to `project/outputs/networks_results/`; resume-validation
histories and other runs are excluded.

| File or field | Meaning |
|---|---|
| Local `report.json` | Authoritative final run status |
| `source-report.json` | Report captured when rendering began; an automatic run may still show RUNNING here |
| Visualization `manifest.json` | Input/export hashes, actual plotting package versions, counts, and transformations |
| `delivery_roots` | Automatic local data/figure roots, relative to the local project |
| File `group` and `path` | `data` or `figure`, with a path relative to that delivery root |
| `render_output_dir` | Actual rendering directory on haiqin |
| `purpose` | `engineering_validation` for smoke; no-accuracy-threshold purpose for selected/experiment runs |

The exact selected-run purpose is
`configured_single_model_without_accuracy_acceptance_threshold`; experiment
uses `configured_experiment_without_accuracy_acceptance_threshold`. A PASS
report confirms the configured pipeline, without a scientific accuracy threshold.
