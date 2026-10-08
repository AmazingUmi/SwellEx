# Result visualization

`scripts/visualization/plot_results.py` renders a single recorded run. Automatic
smoke/experiment runs create figures after successful model evaluation. Use the
[remote workflow](remote-validation.md) for dispatch, environment setup, and
retrieval.

## Result and Provenance Contract

The local delivery directories are:

```text
output/data/<run_id>/
  report.json, manifest.json, request.json, environment.json
  logs/                             exact commands and train/test logs
  reports/                          checks and metrics
  reports/visualization/            metrics.csv, source-report.json, manifest.json
  project/outputs/networks_results/  original prediction CSVs and history JSONs
output/figure/<run_id>/
  range-time.png / .pdf
  prediction-scatter.png / .pdf
  residuals-time.png / .pdf
  absolute-error-ecdf.png / .pdf
  error-metrics.png / .pdf
  training/<method-model>.png / .pdf
  grnn/<method-model>-sigma.png / .pdf
  models/                           original per-model prediction plots
```

Datasets and checkpoints stay on haiqin under the run's `project/outputs/`.
Neither `output/` nor `outputs/` is committed to Git. The visualization manifest
records every input hash, actual package versions, exported file hashes,
sample counts, and transformations. Each file has a `group` (`data` or `figure`)
and a path relative to the corresponding `delivery_roots` directory. These roots
describe automatic workflow delivery relative to the local project;
`render_output_dir` records the actual rendering directory on haiqin.
`source-report.json` captures the report
as read by the renderer; during an automatic run its top-level status can still
be RUNNING, although the selected model evaluation phases have already passed.
The final run status belongs to `output/data/<run_id>/report.json`.

## Figures and units

| Figure | Meaning |
|---|---|
| Range versus time | All label and prediction points; center time in s, range in km |
| Prediction versus label | Common x/y scales, with the ideal `y=x` reference |
| Signed residual versus time | `prediction − label`, in km; common symmetric limits |
| Absolute-error ECDF | Every absolute error, sorted; cumulative rank divided by n |
| Error metrics | MAE, RMSE, maximum absolute error in km, starting at zero |
| Training history | Original loss, MAE and RMSE for each epoch; train/validation identified |
| GRNN sigma search | CV mean or held-out validation metrics; minimum MAPE sigma marked |

GRNN has no training epochs. Its sigma search is a separate figure. SmoothL1
loss is labeled with its configured `km` or `normalized` loss space, and is
not presented as squared distance. No runs are pooled, no error bars are
invented, and no observations are smoothed or downsampled. Prediction CSVs
contain valid test windows only; plots use points rather than connecting
unknown gaps. A periodic test split's spacing does not imply missing predictions.

Smoke figures carry the `engineering_validation` purpose. They demonstrate a
working pipeline; the short data interval and two epochs do not establish
scientific accuracy or model superiority.

## Render or select models manually

Use `SWELLEX_PYTHON` from the
[remote workflow](remote-validation.md#manual-commands-in-a-frozen-run), inside
the selected run's `project/` directory on haiqin:

```bash
"$SWELLEX_PYTHON" -B scripts/visualization/plot_results.py \
  --run "$SWELLEX_RUN" \
  --output-dir output/figure/<run_id>-selected \
  --model RBD/complex_cnn_range --model SCM/scm_grnn_range
```

Repeat `--model METHOD/MODEL` to select entries from the report. With no
selection, all report models are shown. Default formats are PNG (300 dpi) and
PDF; `--formats png` and `--dpi 200` change export settings. The default output
directory is `output/figure/<run_id>/`. Existing destinations are rejected;
choose a new directory for a second rendering. Additional manual outputs are
not automatically fetched by the already completed run. Standalone rendering
writes figures and their CSV/JSON provenance together to the selected directory;
the automatic workflow separates them into the two local delivery roots above.

The public CLI requires a PASS smoke/experiment report. It locates exactly one
prediction CSV and, for CNNs, one history file per model. Ambiguous result
files, malformed/nonfinite values, changed loss space, and inconsistent
reported errors or metrics are rejected. Multiple models must match every
source window index, timestamp, and true label. Discovery is limited to
`project/outputs/networks_results/`, so resume-validation histories and other
runs are excluded. `quick` tests visualization integrity and exports with
synthetic results; it does not create scientific result figures.
