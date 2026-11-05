# Result Figures and Manual Plotting

Selected-model, smoke, and experiment runs automatically render PDFs on `haiqin` and verify retrieved files via `delivery-verification.json`.

## Find Delivered Results

```text
output/data/<run_id>/
  report.json, request.json, manifest.json, environment.json, delivery-verification.json
  logs/                               commands and train/predict logs
  reports/visualization/              metrics.csv, source-report.json, manifest.json
  project/outputs/networks_results/   prediction CSVs and training history JSONs
output/figure/<run_id>/
  range-time.pdf, prediction-scatter.pdf, residuals-time.pdf, absolute-error-ecdf.pdf, error-metrics.pdf
  models/<METHOD>/<model>/{training/history.pdf, test_outputs/prediction.pdf}
  grnn/<method-model>-sigma.pdf
```

Runs from archived packs (ELM, joint-feature, data-scale, MoE) are filed under `archive/output/{data,figure}/<category>/`; see [archive index](../../archive/README.md).

## Choose a Figure

| Figure | Contents and units |
|---|---|
| `range-time.pdf` | Predicted and true range (km) vs window center time (s); unsmoothed points |
| `prediction-scatter.pdf` / `residuals-time.pdf` | Shared-axis `y=x` scatter; signed error (km) vs time with symmetric limits |
| `absolute-error-ecdf.pdf` / `error-metrics.pdf` | Empirical CDF of absolute errors; MAE, RMSE, and max error bars (km) |
| `models/.../history.pdf` / `grnn/...-sigma.pdf` | Neural epoch loss/MAE/RMSE curves; GRNN CV/validation MAPE vs sigma |

`base_pack` grids arrange RBD/SCM rows by architecture columns (`complex_CNN`, `ResNet18`, `ResNet50`, `GRNN`).

## Render Manually on haiqin

First [enter a completed frozen run](remote-environment.md#manual-commands-in-a-frozen-run), then render all or selected models into a **new** directory:

```bash
"$SWELLEX_PYTHON" -B scripts/remote_workflow/visualization/plot_results.py \
  --run "$SWELLEX_RUN" --output-dir output/figure/manual-comparison \
  --model RBD/complex_cnn_range --model SCM/scm_grnn_range
```

| Option | Default | Meaning |
|---|---|---|
| `--run PATH` | Required | Remote run directory with a PASS selected/smoke/experiment `report.json` |
| `--output-dir PATH` | `output/figure/<run_id>/` | Must not exist |
| `--model METHOD/MODEL` | All planned models | Repeatable filter; full-suite checks still run first |
| `--dpi N` | `300` | Embedded raster DPI (`72..1200`) |

## Input Checks and Provenance

Before filtering, the renderer verifies plan completeness, PASS status, exactly one CSV per model and one history per neural model, finite consistent metrics, and identical test window indices/times/labels across models. Visualization `manifest.json` records input/output SHA256 hashes and package versions; pipeline PASS implies no scientific accuracy threshold.
