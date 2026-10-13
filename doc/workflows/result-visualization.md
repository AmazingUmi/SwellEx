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

`range-time`, `prediction-scatter`, and `residuals-time` place each feature
method in its own row, ordered RBD, ELM, SCM. Shared model columns follow
complex CNN, real CNN, ResNet18, ResNet50, GRNN; the GRNN column is populated
for RBD, ELM, and SCM in new complete runs. Model selection omits unused rows/columns and hides missing
feature/model panels while retaining the shared scales.

A selected run has five aggregate prediction/error figures, one per-model
prediction figure, and its neural history or GRNN search figure when applicable.
GRNN has no epoch history. All figures are PDF; no PNG files are saved.
The automatic renderer creates model predictions from the recorded CSV and
training curves from the original history JSON.

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
"$SWELLEX_PYTHON" -B scripts/base_pack/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/manual-all
```

To compare specific models from a completed **smoke or experiment** run,
choose another new directory and repeat `--model`:

```bash
"$SWELLEX_PYTHON" -B scripts/base_pack/visualization/plot_results.py \
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
| `--dpi N` | `300` | Resolution for any embedded raster content in PDF, integer from 72 to 1200 |

`quick` runs test plotting with synthetic results; they do not produce
scientific result figures.

## Input Checks and Provenance

Before rendering, the public CLI checks:

- The recorded complete model plan: six models for smoke, fifteen for
  experiment, or the single selected model. Unknown or missing models fail.
  Historical four/thirteen-model reports are accepted only with the known
  SCM-only source-plan identity; this does not relax new report completeness.
- Passed evaluation phases for every model in that plan.
- Exactly one prediction CSV per model, and one history file per neural model.
- Finite, well-formed rows and consistent recorded errors, metrics, and loss space.
- Matching source-window indices, timestamps, and true labels across all run models.
- For `selected`, exactly one model metric and a matching canonical
  `method`/`architecture`/`model` selection.

These checks run before `--model` filtering. Selection controls rendering;
missing or damaged inputs for unselected models still fail, and input provenance
includes the files verified for the full run.

Discovery is limited to `project/outputs/networks_results/`; resume-validation
histories and other runs are excluded.

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

Fetch verifies each listed file's safe relative path, size, and SHA-256 after
checksum retrieval. Its recorded scope is `visualization_manifest_files`:
the rendering manifest's exported PDFs (including all per-model prediction and
training PDFs), metrics, and captured source report.
Other exported logs use rsync checksum retrieval and are outside this visualization
manifest. Renderer source-file hashes are recorded separately from the training
source snapshot.
A PASS model run with a missing rendering manifest fails delivery verification;
its remote computation status remains recorded separately in `report.json`.

The exact selected-run purpose is
`configured_single_model_without_accuracy_acceptance_threshold`; experiment
uses `configured_experiment_without_accuracy_acceptance_threshold`. A PASS
report confirms the configured pipeline, without a scientific accuracy threshold.
