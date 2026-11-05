# RBD/SCM GRNN Build and Prediction

`scripts/network_training/grnn/` implements Gaussian-kernel regression via `build` (store reference samples/sigma) and `predict`, with no epochs or optimizer. Equations are in the [GRNN method note](grnn-method.md); remote workflow runs use `./scripts/run.sh base_pack --method SCM --network GRNN` (or `local-grnn` in [remote validation](../remote_workflow/run-and-validate.md#maintenance-commands)).

## Manual Build and Predict

[Enter a frozen run](../remote_workflow/remote-environment.md#manual-commands-in-a-frozen-run) on `haiqin` and choose a dataset directory from `report.json`:

```bash
GRNN_OUTPUT='outputs/manual_results/grnn'
"$SWELLEX_PYTHON" -B scripts/network_training/grnn/main.py build \
  --method RBD --data '<dataset_name>' --device cuda --cv-folds 3 \
  --sigma-candidates 0.01 0.1 1.0 --batch-size 64 --seed 2026 --output-dir "$GRNN_OUTPUT"

"$SWELLEX_PYTHON" -B scripts/network_training/grnn/main.py predict \
  --method RBD --data '<dataset_name>' --device cuda \
  --reference '<path_to_reference>' --batch-size 64 --output-dir "$GRNN_OUTPUT"
```

Shared manual CLI options: `--method RBD|SCM` (required), `--data DATASET` (name under `outputs/Datasets/` or directory path), `--device` (default CUDA if available), `--output-dir` (default `outputs/networks_results/<METHOD>/`), and `--batch-size` (default `128`; workflow TOML uses `1024`).

## Build Options

| Option | Default | Meaning |
|---|---|---|
| `--train-data GLOB ...` / `--val-data GLOB ...` | `*_train.h5` / None | Reference files and optional held-out validation files |
| `--spread VALUE` | `0.01` | Fixed sigma when neither `--val-data` nor `--cv-folds > 1` is active |
| `--cv-folds N` / `--seed N` | `0` / `2026` | Reference k-fold CV when `N >= 2`; CV shuffle seed |
| `--sigma-candidates VALUE ...` | `0.01..0.10`, `0.20..2.00` | Positive candidates with finite normal float32 `2*sigma^2` |
| `--standardize-input` / `--per-sample-channel-norm` | Off / Off | Per-coordinate z-score on reference; per-sample real/imag z-score |
| `--max-reference-samples N` / `--max-validation-samples N` | No limit | Optional positive debugging caps |

**Sigma selection order**: (1) explicit `--val-data` minimizes held-out MAPE; (2) `--cv-folds > 1` minimizes mean fold MAPE; (3) otherwise fixed `--spread` (`*_val.h5` is never auto-loaded without `--val-data`).

## Predict Options and Artifacts

| Option | Default | Meaning |
|---|---|---|
| `--reference PATH` | `<dataset>_reference.pt`, else latest `MMDD_HHMMSS` | Explicit missing path fails; physical signature must match |
| `--input-data GLOB ...` / `--max-predict-samples N` | `*_test.h5` / No limit | Test files and optional cap |
| `--predictions-csv PATH` / `--plot-path PATH` / `--no-plot` | Timestamped CSV & `.pdf` / Off | Output overrides (`--print-first` defaults to `10`) |

Flattened real dimensions under `outputs/networks_results/<METHOD>/<method>_grnn_range/` are `2*E*F` (RBD) and `E*(E+1)*F` (SCM), writing `reference_outputs/<dataset>_{reference,sigma_search}_MMDD_HHMMSS.{pt,json}` and `test_outputs/<dataset>_predictions_MMDD_HHMMSS.csv`.
